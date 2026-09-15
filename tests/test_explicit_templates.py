# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
from pathlib import Path

import numpy as np
import pytest
from biotite.structure import AtomArray

from opendde.data.template.template_featurizer import InferenceTemplateFeaturizer
from opendde.data.template.template_parser import TemplateHit, TemplateSearchResult
from opendde.data.template.template_utils import TemplateHitFeaturizer


def _cif(chains=("A",)):
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


def test_explicit_template_requires_single_chain_without_chain_id(tmp_path):
    from opendde.data.template.template_finalizer import load_explicit_template_features

    (tmp_path / "tiny.cif").write_text(_cif(("A", "B")))
    processor = TemplateHitFeaturizer(mmcif_dir=str(tmp_path))._hit_processor
    entry = {
        "mmcifPath": "tiny.cif",
        "queryIndices": list(range(7)),
        "templateIndices": list(range(7)),
    }
    with pytest.raises(ValueError, match="chainId"):
        load_explicit_template_features(
            "AAAAAAA", [entry], base_dir=tmp_path, template_processor=processor
        )
    entry["chainId"] = "B"
    features = load_explicit_template_features(
        "AAAAAAA", [entry], base_dir=tmp_path, template_processor=processor
    )
    assert features[0]["template_all_atom_masks"].sum() == 7
    assert features[0]["template_domain_names"].item() == b"tiny_B"


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
    assert Path(entry["mmcifPath"]).read_text() == text
    if len(chains) > 1:
        assert entry["chainId"] == "B"
    else:
        assert set(entry) == {"mmcifPath", "queryIndices", "templateIndices"}


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
