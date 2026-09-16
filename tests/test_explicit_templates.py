# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import json
from pathlib import Path

import numpy as np
import pytest
from biotite.structure import AtomArray
from ml_collections import ConfigDict

from opendde.data.template.template_featurizer import InferenceTemplateFeaturizer
from opendde.data.template.template_parser import (
    TemplateHit,
    TemplateParser,
    TemplateSearchResult,
)
from opendde.data.template.template_utils import TemplateHitFeaturizer
from opendde.utils.text_io import write_zstd_text_atomic


def _cif(chains=("A",), *, missing_residues=None):
    missing_residues = missing_residues or {}
    header = """data_tiny
_entry.id tiny
_pdbx_audit_revision_history.revision_date 2099-01-01
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
"""
    header += "".join(f"1 {i} ALA\n" for i in range(1, 8))
    header += """loop_
_chem_comp.id
_chem_comp.type
ALA 'L-peptide linking'
loop_
_struct_asym.id
_struct_asym.entity_id
"""
    header += "".join(f"{chain} 1\n" for chain in chains)
    header += """loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
"""
    for chain_index, chain in enumerate(chains):
        for i in range(1, 8):
            if i in missing_residues.get(chain, ()):
                continue
            header += (
                f"ATOM {chain_index * 7 + i} C CA . ALA {chain} 1 {i} ? "
                f"{i * 3.8} {chain_index * 10} 0 1 20 {i} {chain} 1\n"
            )
    return header


def _atoms():
    atoms = AtomArray(7)
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = np.arange(1, 8)
    atoms.set_annotation("asym_id_int", np.zeros(7, dtype=int))
    atoms.set_annotation("centre_atom_mask", np.ones(7, dtype=bool))
    return atoms


def _online(tmp_path, monkeypatch):
    featurizer = TemplateHitFeaturizer(
        mmcif_dir=str(tmp_path), max_hits=4, max_template_date="2000-01-01"
    )
    monkeypatch.setattr(
        featurizer,
        "get_templates",
        lambda **kwargs: pytest.fail("Explicit templates must bypass hit filtering"),
    )
    return featurizer


def test_explicit_indices_reach_existing_extractor(tmp_path, monkeypatch):
    from opendde.data.template.template_finalizer import load_explicit_template_features

    (tmp_path / "tiny.cif").write_text(_cif())
    online = _online(tmp_path, monkeypatch)

    def extract(
        mmcif_obj, pdb_id, mapping, template_seq, query_seq, chain_id, **kwargs
    ):
        assert mapping == {0: 5, 1: 6}
        assert template_seq == "AAAAAAA"
        assert query_seq == "AA"
        assert chain_id == "A"
        assert mmcif_obj.header["release_date"] == "2099-01-01"
        return {"template_sequence": b"AA"}, None

    monkeypatch.setattr(online._hit_processor, "_extract_template_features", extract)
    features = load_explicit_template_features(
        "AA",
        [{"mmcifPath": "tiny.cif", "queryIndices": [0, 1], "templateIndices": [5, 6]}],
        base_dir=tmp_path,
        template_processor=online._hit_processor,
    )
    assert features[0]["template_sequence"] == b"AA"
    assert features[0]["template_sum_probs"] == [0.0]
    assert features[0]["template_release_date"].item() == b"2099-01-01"


def test_explicit_template_reads_zstd_by_magic(tmp_path, monkeypatch):
    from opendde.data.template.template_finalizer import load_explicit_template_features

    write_zstd_text_atomic(tmp_path / "tiny.cif.zst", _cif())
    online = _online(tmp_path, monkeypatch)
    monkeypatch.setattr(
        online._hit_processor,
        "_extract_template_features",
        lambda **_kwargs: ({"template_sequence": b"AA"}, None),
    )

    features = load_explicit_template_features(
        "AA",
        [
            {
                "mmcifPath": "tiny.cif.zst",
                "queryIndices": [0, 1],
                "templateIndices": [0, 1],
            }
        ],
        base_dir=tmp_path,
        template_processor=online._hit_processor,
    )

    assert features[0]["template_sequence"] == b"AA"


@pytest.mark.parametrize("empty", [False, True])
def test_explicit_templates_override_legacy_hits_and_dates(
    tmp_path, monkeypatch, empty
):
    path = tmp_path / "tiny.cif"
    path.write_text(_cif())
    legacy = tmp_path / "hits.invalid"
    legacy.write_text("must not parse")
    entry = {
        "mmcifPath": str(path),
        "queryIndices": list(range(7)),
        "templateIndices": list(range(7)),
    }
    result = InferenceTemplateFeaturizer.make_template_feature(
        [
            {
                "proteinChain": {
                    "sequence": "AAAAAAA",
                    "count": 1,
                    "templates": [] if empty else [entry],
                    "templatesPath": str(legacy),
                }
            }
        ],
        _atoms(),
        online_template_featurizer=_online(tmp_path, monkeypatch),
    )
    assert result["template_atom_mask"].shape[:2] == (4, 7)
    assert int(result["template_atom_mask"].sum()) == (0 if empty else 7)


def test_explicit_template_requires_single_chain_and_rejects_chain_id(tmp_path):
    from opendde.data.template.template_finalizer import load_explicit_template_features

    (tmp_path / "tiny.cif").write_text(_cif(("A", "B")))
    processor = TemplateHitFeaturizer(mmcif_dir=str(tmp_path))._hit_processor
    entry = {
        "mmcifPath": "tiny.cif",
        "queryIndices": list(range(7)),
        "templateIndices": list(range(7)),
    }
    with pytest.raises(ValueError, match="single protein chain"):
        load_explicit_template_features(
            "AAAAAAA", [entry], base_dir=tmp_path, template_processor=processor
        )

    (tmp_path / "single.cif").write_text(_cif())
    entry["mmcifPath"] = "single.cif"
    entry["chainId"] = "A"
    with pytest.raises(ValueError, match="does not support chainId"):
        load_explicit_template_features(
            "AAAAAAA", [entry], base_dir=tmp_path, template_processor=processor
        )


def test_explicit_without_release_date_works_offline(tmp_path):
    path = tmp_path / "undated.cif"
    path.write_text(
        _cif().replace("_pdbx_audit_revision_history.revision_date 2099-01-01\n", "")
    )
    result = InferenceTemplateFeaturizer.make_template_feature(
        [
            {
                "proteinChain": {
                    "sequence": "AAAAAAA",
                    "count": 1,
                    "templates": [
                        {
                            "mmcifPath": "undated.cif",
                            "queryIndices": list(range(7)),
                            "templateIndices": list(range(7)),
                        }
                    ],
                }
            }
        ],
        _atoms(),
        base_dir=tmp_path,
    )
    assert result["template_atom_mask"].sum() == 7


@pytest.mark.parametrize("chains", [("A",), ("A", "B")])
def test_finalize_serializes_selected_realigned_mapping(tmp_path, monkeypatch, chains):
    from opendde.data.template.template_finalizer import finalize_template_hits

    selected_chain = chains[-1]
    text = _cif(chains)
    (tmp_path / "tiny.cif").write_text(text)
    hits_path = tmp_path / "hits.a3m"
    hits_path.write_text(f">tiny_{selected_chain}/1-7 mol:protein length:7\nAAAAAAA\n")
    online = TemplateHitFeaturizer(mmcif_dir=str(tmp_path), max_hits=4)
    calls = []

    def get_templates(**kwargs):
        calls.append(kwargs)
        hit = TemplateHit(
            1,
            f"tiny_{selected_chain}",
            2,
            None,
            "AAAAAAA",
            "AAAAAAA",
            [0, 1, 2],
            [5, -1, 6],
        )
        return TemplateSearchResult([{}], [hit], [], []), {}

    monkeypatch.setattr(online, "get_templates", get_templates)
    entries = finalize_template_hits(
        "AAAAAAA", hits_path, online, max_template_date="2000-01-01"
    )
    assert len(calls) == 1
    assert calls[0]["max_template_date"] == "2000-01-01"
    assert len(calls[0]["hits"]) == 1
    assert len(entries) == 1
    entry = entries[0]
    assert entry["queryIndices"] == [0, 2]
    assert entry["templateIndices"] == [5, 6]
    assert set(entry) == {"mmcifPath", "queryIndices", "templateIndices"}
    assert Path(entry["mmcifPath"]).name == f"tiny_{selected_chain}.cif"
    extracted = TemplateParser.parse(
        file_id="tiny", mmcif_string=Path(entry["mmcifPath"]).read_text()
    ).mmcif_object
    assert extracted is not None
    assert extracted.chain_to_seqres == {selected_chain: "AAAAAAA"}


def test_finalized_single_chain_preserves_unresolved_residues_and_features(
    tmp_path, monkeypatch
):
    from opendde.data.template.template_finalizer import (
        finalize_template_hits,
        load_explicit_template_features,
    )
    from opendde.data.template.template_parser import TemplateParser

    text = _cif(("A", "B"), missing_residues={"B": {4}})
    (tmp_path / "tiny.cif").write_text(text)
    hits_path = tmp_path / "hits.a3m"
    hits_path.write_text(">tiny_B/1-7 mol:protein length:7\nAAAAAAA\n")
    online = TemplateHitFeaturizer(mmcif_dir=str(tmp_path), max_hits=4)
    hit = TemplateHit(
        1,
        "tiny_B",
        7,
        None,
        "AAAAAAA",
        "AAAAAAA",
        list(range(7)),
        list(range(7)),
    )
    monkeypatch.setattr(
        online,
        "get_templates",
        lambda **_kwargs: (TemplateSearchResult([{}], [hit], [], []), {}),
    )

    [entry] = finalize_template_hits(
        "AAAAAAA", hits_path, online, max_template_date="2000-01-01"
    )
    original = TemplateParser.parse(
        file_id="tiny", mmcif_string=text, auth_chain_id="B"
    ).mmcif_object
    extracted = TemplateParser.parse(
        file_id="tiny", mmcif_string=Path(entry["mmcifPath"]).read_text()
    ).mmcif_object
    assert original is not None and extracted is not None
    assert original.chain_to_seqres == extracted.chain_to_seqres == {"B": "AAAAAAA"}
    assert original.seqres_to_structure["B"][3].is_missing
    assert extracted.seqres_to_structure["B"][3].is_missing

    processor = online._hit_processor
    mapping = dict(zip(entry["queryIndices"], entry["templateIndices"], strict=True))
    original_features, _ = processor._extract_template_features(
        mmcif_obj=original,
        pdb_id="tiny",
        mapping=mapping,
        template_seq=original.chain_to_seqres["B"],
        query_seq="AAAAAAA",
        chain_id="B",
        _zero_center=processor._zero_center_positions,
    )
    original_features["template_sum_probs"] = [0.0]
    original_features["template_release_date"] = np.array(b"2099-01-01", dtype=object)
    [extracted_features] = load_explicit_template_features(
        "AAAAAAA",
        [entry],
        base_dir=tmp_path,
        template_processor=processor,
    )
    assert original_features.keys() == extracted_features.keys()
    for key in original_features:
        np.testing.assert_array_equal(original_features[key], extracted_features[key])


def test_update_finalizes_existing_hits_without_search(tmp_path, monkeypatch):
    from runner import template_search

    path = tmp_path / "hits.a3m"
    path.write_text("")
    online = TemplateHitFeaturizer(mmcif_dir=str(tmp_path), max_hits=4)
    monkeypatch.setattr(
        online,
        "get_templates",
        lambda **kwargs: (TemplateSearchResult([], [], [], []), {}),
    )
    jobs = [
        {
            "sequences": [
                {"proteinChain": {"sequence": "AAAAAAA", "templatesPath": str(path)}}
            ]
        }
    ]
    assert template_search.update_template_info(
        jobs, template_featurizer=online, max_template_date="2000-01-01"
    )
    assert jobs[0]["sequences"][0]["proteinChain"] == {
        "sequence": "AAAAAAA",
        "templates": [],
    }


def test_update_leaves_explicit_empty_templates_untouched():
    from runner.template_search import update_template_info

    jobs = [{"sequences": [{"proteinChain": {"sequence": "AAAAAAA", "templates": []}}]}]
    assert not update_template_info(jobs)


def _dataset_config(tmp_path, *, fetch_remote=False):
    return ConfigDict(
        {
            "input_json_path": str(tmp_path / "job_data.json"),
            "dump_dir": str(tmp_path / "out"),
            "use_msa": False,
            "use_template": True,
            "data": {
                "ccd_components_file": None,
                "ccd_components_rdkit_mol_file": None,
                "template": {
                    "prot_template_mmcif_dir": str(tmp_path / "missing_database"),
                    "prot_template_cache_dir": None,
                    "kalign_binary_path": None,
                    "release_dates_path": None,
                    "obsolete_pdbs_path": None,
                    "fetch_remote": fetch_remote,
                },
            },
        }
    )


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("fetch_remote", [False, True])
def test_dataset_explicit_templates_need_no_database_or_online_featurizer(
    tmp_path, empty, fetch_remote
):
    """Prepared local mappings must work even with remote fetching disabled."""
    from opendde.data.inference.infer_dataloader import InferenceDataset

    path = tmp_path / "tiny.cif"
    path.write_text(_cif())
    sequences = [
        {
            "proteinChain": {
                "sequence": "AAAAAAA",
                "count": 1,
                "templates": []
                if empty
                else [
                    {
                        "mmcifPath": str(path),
                        "queryIndices": list(range(7)),
                        "templateIndices": list(range(7)),
                    }
                ],
                "templatesPath": "ignored_legacy.a3m",
            }
        }
    ]
    configs = _dataset_config(tmp_path, fetch_remote=fetch_remote)
    Path(configs.input_json_path).write_text(
        json.dumps([{"name": "job", "sequences": sequences}])
    )
    dataset = InferenceDataset(configs)
    assert dataset.online_template_featurizer is None
    assert not (tmp_path / "missing_database").exists()
    result = InferenceTemplateFeaturizer.make_template_feature(
        sequences,
        _atoms(),
        online_template_featurizer=dataset.online_template_featurizer,
    )
    assert int(result["template_atom_mask"].sum()) == (0 if empty else 7)


def test_dataset_mixed_templates_still_requires_and_constructs_legacy_featurizer(
    tmp_path,
):
    from opendde.data.inference.infer_dataloader import InferenceDataset

    inputs = [
        {
            "name": "job",
            "sequences": [
                {"proteinChain": {"sequence": "AAAAAAA", "templates": []}},
                {"proteinChain": {"sequence": "AAAAAAA", "templatesPath": "hits.a3m"}},
            ],
        }
    ]
    configs = _dataset_config(tmp_path)
    with pytest.raises(AssertionError, match="mmcif directory"):
        InferenceDataset(configs, inputs=inputs)
    (tmp_path / "missing_database").mkdir()
    dataset = InferenceDataset(configs, inputs=inputs)
    assert isinstance(dataset.online_template_featurizer, TemplateHitFeaturizer)
