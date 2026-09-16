# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

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
        path = Path(base_dir) / entry["mmcifPath"]
        logical_path = path.with_suffix("") if path.suffix == ".zst" else path
        chain_id = entry.get("chainId")
        parsed = TemplateParser.parse(
            file_id=logical_path.stem,
            mmcif_string=read_text(path),
            auth_chain_id=chain_id,
        )
        mmcif = parsed.mmcif_object
        if mmcif is None or not mmcif.chain_to_seqres:
            raise ValueError(
                f"Could not parse protein template {path}: {parsed.errors}"
            )
        if chain_id is None:
            if len(mmcif.chain_to_seqres) != 1:
                raise ValueError(
                    f"Template {path} requires chainId for multiple protein chains"
                )
            chain_id = next(iter(mmcif.chain_to_seqres))
        mapping = dict(
            zip(entry["queryIndices"], entry["templateIndices"], strict=True)
        )
        hit_features, _ = template_processor._extract_template_features(
            mmcif_obj=mmcif,
            pdb_id=logical_path.stem,
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
        cif_path = path.parent / "templates" / f"{pdb_id}.cif"
        cif_path.parent.mkdir(parents=True, exist_ok=True)
        cif_path.write_text(cif_text)
        mapping = hit.query_to_hit_mapping
        entry = {
            "mmcifPath": str(cif_path.resolve()),
            "queryIndices": list(mapping),
            "templateIndices": list(mapping.values()),
        }
        if len(mmcif.chain_to_seqres) > 1:
            entry["chainId"] = chain_id
        entries.append(entry)
    return entries
