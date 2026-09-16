# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
from datetime import datetime
import io
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from Bio import PDB

from opendde.data.template.template_parser import (
    HHRParser,
    HmmsearchA3MParser,
    TemplateParser,
    get_pdb_id_and_chain,
)
from opendde.data.template.template_utils import (
    TemplateHitFeaturizer,
    TemplateHitProcessor,
)
from opendde.utils.text_io import read_text


def _filtered_loop(
    source: Mapping[str, Sequence[str]], prefix: str, indices: Sequence[int]
) -> dict[str, list[str]]:
    return {
        key: [values[index] for index in indices]
        for key, values in source.items()
        if key.startswith(prefix)
    }


def _single_chain_mmcif(mmcif: Any, auth_chain_id: str) -> str:
    """Serialize one protein chain while retaining its full polymer sequence."""
    source = mmcif.raw_string
    protein_label_chains = TemplateParser._get_protein_chains(source)
    selected_label_chains = {
        label_chain
        for label_chain, author_chain in zip(
            source["_atom_site.label_asym_id"],
            source["_atom_site.auth_asym_id"],
            strict=True,
        )
        if author_chain == auth_chain_id and label_chain in protein_label_chains
    }
    if len(selected_label_chains) != 1:
        raise ValueError(
            f"Expected one protein chain for {auth_chain_id}, found "
            f"{len(selected_label_chains)}"
        )
    label_chain_id = next(iter(selected_label_chains))

    atom_indices = [
        index
        for index, label_chain in enumerate(source["_atom_site.label_asym_id"])
        if label_chain == label_chain_id
    ]
    entity_ids = {source["_atom_site.label_entity_id"][index] for index in atom_indices}
    struct_asym_indices = [
        index
        for index, label_chain in enumerate(source["_struct_asym.id"])
        if label_chain == label_chain_id
    ]
    entity_poly_seq_indices = [
        index
        for index, entity_id in enumerate(source["_entity_poly_seq.entity_id"])
        if entity_id in entity_ids
    ]
    monomer_ids = {
        source["_entity_poly_seq.mon_id"][index] for index in entity_poly_seq_indices
    }

    output: dict[str, Any] = {
        "data_": f"{mmcif.file_id}_{auth_chain_id}",
    }
    metadata_prefixes = (
        "_entry.",
        "_exptl.",
        "_pdbx_audit_revision_history.",
        "_refine.",
        "_em_3d_reconstruction.",
        "_reflns.",
    )
    for key, values in source.items():
        if key.startswith(metadata_prefixes):
            output[key] = list(values)

    output.update(_filtered_loop(source, "_atom_site.", atom_indices))
    output.update(_filtered_loop(source, "_struct_asym.", struct_asym_indices))
    output.update(_filtered_loop(source, "_entity_poly_seq.", entity_poly_seq_indices))

    for prefix, id_key in (
        ("_entity.", "_entity.id"),
        ("_entity_poly.", "_entity_poly.entity_id"),
    ):
        if id_key in source:
            indices = [
                index
                for index, entity_id in enumerate(source[id_key])
                if entity_id in entity_ids
            ]
            output.update(_filtered_loop(source, prefix, indices))

    if "_entity_poly.pdbx_strand_id" in output:
        output["_entity_poly.pdbx_strand_id"] = [auth_chain_id]

    if "_chem_comp.id" in source:
        chem_comp_indices = [
            index
            for index, monomer_id in enumerate(source["_chem_comp.id"])
            if monomer_id in monomer_ids
        ]
        output.update(_filtered_loop(source, "_chem_comp.", chem_comp_indices))

    if "_pdbx_poly_seq_scheme.asym_id" in source:
        scheme_indices = [
            index
            for index, label_chain in enumerate(source["_pdbx_poly_seq_scheme.asym_id"])
            if label_chain == label_chain_id
        ]
        output.update(_filtered_loop(source, "_pdbx_poly_seq_scheme.", scheme_indices))

    writer = PDB.MMCIFIO()
    writer.set_dict(output)
    buffer = io.StringIO()
    writer.save(buffer)
    return buffer.getvalue()


def load_explicit_template_features(
    query_sequence: str,
    templates: Sequence[Mapping[str, Any]],
    *,
    base_dir: str | Path,
    template_processor: TemplateHitProcessor,
) -> list[Mapping[str, Any]]:
    """Extract features from explicit zero-based mappings, without date filtering."""
    features = []
    for entry in templates:
        if "chainId" in entry:
            raise ValueError(
                "An explicit template does not support chainId; provide a single-chain "
                "mmCIF file"
            )
        path = Path(base_dir) / entry["mmcifPath"]
        logical_path = path.with_suffix("") if path.suffix == ".zst" else path
        parsed = TemplateParser.parse(
            file_id=logical_path.stem,
            mmcif_string=read_text(path),
        )
        mmcif = parsed.mmcif_object
        if mmcif is None or not mmcif.chain_to_seqres:
            raise ValueError(
                f"Could not parse protein template {path}: {parsed.errors}"
            )
        if len(mmcif.chain_to_seqres) != 1:
            raise ValueError(f"Template {path} must be a single protein chain")
        chain_id = next(iter(mmcif.chain_to_seqres))
        entry_ids = mmcif.raw_string.get("_entry.id", [])
        template_id = entry_ids[0] if entry_ids else logical_path.stem
        if template_id in (".", "?"):
            template_id = logical_path.stem
        mapping = dict(
            zip(entry["queryIndices"], entry["templateIndices"], strict=True)
        )
        hit_features, _ = template_processor._extract_template_features(
            mmcif_obj=mmcif,
            pdb_id=template_id,
            mapping=mapping,
            template_seq=mmcif.chain_to_seqres[chain_id],
            query_seq=query_sequence,
            chain_id=chain_id,
            _zero_center=template_processor._zero_center_positions,
        )
        hit_features["template_sum_probs"] = [0.0]
        # Assembly requires a parseable date even for user structures without one.
        # This metadata is not used to filter explicit templates.
        hit_features["template_release_date"] = np.array(
            mmcif.header.get("release_date", "1970-01-01").encode(), dtype=object
        )
        features.append(hit_features)
    return features


def finalize_template_hits(
    query_sequence: str,
    templates_path: str | Path,
    template_featurizer: TemplateHitFeaturizer,
    *,
    max_template_date: str | datetime | None,
) -> list[dict[str, Any]]:
    """Freeze selected, realigned search hits as portable explicit templates."""
    path = Path(templates_path)
    content = read_text(path)
    logical_path = path.with_suffix("") if path.suffix == ".zst" else path
    if logical_path.suffix == ".a3m":
        hits = HmmsearchA3MParser.parse(
            query_seq=query_sequence, a3m_str=content, skip_first=False
        )
    elif logical_path.suffix == ".hhr":
        hits = HHRParser.parse(hhr_string=content)
    else:
        raise ValueError(f"Unsupported template format: {path}")
    result, _ = template_featurizer.get_templates(
        sequence_uid=query_sequence,
        query_sequence=query_sequence,
        hits=hits,
        max_template_date=max_template_date,
    )
    entries = []
    for hit in result.hits[:4]:
        pdb_id, chain_id = get_pdb_id_and_chain(hit)
        pdb_id = template_featurizer._obsolete_pdbs.get(pdb_id, pdb_id)
        cif_text = template_featurizer._hit_processor._fetch_or_read_cif(pdb_id)
        parsed = TemplateParser.parse(file_id=pdb_id, mmcif_string=cif_text)
        mmcif = parsed.mmcif_object
        if mmcif is None or not mmcif.chain_to_seqres:
            raise ValueError(
                f"Could not parse selected template {pdb_id}: {parsed.errors}"
            )
        single_chain_cif = _single_chain_mmcif(mmcif, chain_id)
        cif_path = path.parent / "templates" / f"{pdb_id}_{chain_id}.cif"
        cif_path.parent.mkdir(parents=True, exist_ok=True)
        cif_path.write_text(single_chain_cif)
        mapping = hit.query_to_hit_mapping
        entry = {
            "mmcifPath": str(cif_path.resolve()),
            "queryIndices": list(mapping),
            "templateIndices": list(mapping.values()),
        }
        entries.append(entry)
    return entries
