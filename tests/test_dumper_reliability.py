# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import json
import os
import stat

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray, BondList
from biotite.structure.io import pdbx

from runner.dumper import DataDumper, get_clean_full_confidence


@pytest.fixture
def atom_array(monkeypatch):
    # Keep CIF serialization real without requiring the external CCD database.
    monkeypatch.setattr("opendde.data.utils.biotite_load_ccd_cif", lambda: {})
    atoms = AtomArray(1)
    atoms.atom_name[:] = "CA"
    atoms.res_name[:] = "ALA"
    atoms.res_id[:] = 1
    atoms.chain_id[:] = "A"
    atoms.element[:] = "C"
    atoms.bonds = BondList(1)
    atoms.set_annotation("label_asym_id", np.array(["A"]))
    atoms.set_annotation("label_entity_id", np.array(["1"]))
    return atoms


def _minimal_prediction():
    return {
        "coordinate": torch.zeros(1, 1, 3),
        "summary_confidence": [{"ranking_score": 1.0}],
        "full_data": [{}],
    }


@pytest.mark.parametrize("sorted_by_ranking_score", [True, False])
def test_dump_uses_original_sample_indices_and_preserves_other_seeds(
    tmp_path, atom_array, sorted_by_ranking_score
):
    dumper = DataDumper(
        str(tmp_path),
        need_atom_confidence=True,
        sorted_by_ranking_score=sorted_by_ranking_score,
    )
    prediction = {
        "coordinate": torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]),
        "summary_confidence": [{"ranking_score": 0.1}, {"ranking_score": 0.9}],
        "full_data": [
            {"atom_plddt": torch.tensor([0.25], dtype=torch.bfloat16)},
            {"atom_plddt": torch.tensor([0.75], dtype=torch.bfloat16)},
        ],
    }
    job_dir = tmp_path / "job"
    first_seed_files = {}
    for seed in (7, 8):
        dumper.dump(
            group_name="",
            pdb_id="job",
            seed=seed,
            pred_dict=prediction,
            atom_array=atom_array,
            entity_poly_type={"1": "polypeptide(L)"},
        )
        for index, coordinates, score, plddt in (
            (0, [1.0, 2.0, 3.0], 0.1, 0.25),
            (1, [4.0, 5.0, 6.0], 0.9, 0.75),
        ):
            model = job_dir / "models" / f"seed-{seed}_sample-{index}_model.cif"
            block = pdbx.CIFFile.read(model).block
            assert block["entry"]["id"].as_item() == "job"
            atoms = block["atom_site"]
            actual_coordinates = [
                atoms[column].as_array(float)[0]
                for column in ("Cartn_x", "Cartn_y", "Cartn_z")
            ]
            np.testing.assert_allclose(actual_coordinates, coordinates)
            np.testing.assert_allclose(
                atoms["B_iso_or_equiv"].as_array(float), [plddt * 100]
            )
            summary = (
                job_dir
                / "summary_confidences"
                / (f"seed-{seed}_sample-{index}_summary_confidences.json")
            )
            assert json.loads(summary.read_text()) == {"ranking_score": score}
            full_data = (
                job_dir / "full_data" / f"seed-{seed}_sample-{index}_full_data.json"
            )
            assert json.loads(full_data.read_text()) == {"atom_plddt": [plddt]}
        if seed == 7:
            first_seed_files = {
                path: path.read_bytes() for path in job_dir.rglob("*") if path.is_file()
            }

    assert len(first_seed_files) == 6
    assert all(
        path.read_bytes() == content for path, content in first_seed_files.items()
    )
    assert {path.name for path in job_dir.iterdir()} == {
        "models",
        "summary_confidences",
        "full_data",
    }
    assert len(list(job_dir.rglob("*.cif"))) == 4
    assert len(list(job_dir.rglob("*.json"))) == 8
    assert "b_factor" not in atom_array.get_annotation_categories()
    assert prediction["full_data"][0]["atom_plddt"].dtype == torch.bfloat16


def test_dump_omits_full_data_directory_when_disabled(tmp_path, atom_array):
    dumper = DataDumper(str(tmp_path))
    prediction = _minimal_prediction()
    prediction.pop("full_data")
    dumper.dump(
        group_name="",
        pdb_id="job",
        seed=1,
        pred_dict=prediction,
        atom_array=atom_array,
        entity_poly_type={"1": "polypeptide(L)"},
    )

    assert {
        path.relative_to(tmp_path / "job").as_posix()
        for path in (tmp_path / "job").rglob("*")
        if path.is_file()
    } == {
        "models/seed-1_sample-0_model.cif",
        "summary_confidences/seed-1_sample-0_summary_confidences.json",
    }
    assert not (tmp_path / "job" / "full_data").exists()


def test_confidence_serialization_does_not_mutate_prediction_tree():
    nested = torch.tensor([1.234], dtype=torch.float32)
    confidence = {
        "atom_coordinate": torch.ones(1, 3),
        "atom_is_polymer": torch.ones(1, dtype=torch.bool),
        "nested": {"score": nested},
        "values": np.array([2.345]),
    }

    cleaned = get_clean_full_confidence(confidence)

    assert "atom_coordinate" in confidence
    assert "atom_is_polymer" in confidence
    assert confidence["nested"]["score"] is nested
    assert "atom_coordinate" not in cleaned
    assert "atom_is_polymer" not in cleaned
    np.testing.assert_allclose(cleaned["nested"]["score"], np.array([1.23]))
    np.testing.assert_allclose(cleaned["values"], np.array([2.35]))


def test_confidence_cleaning_preserves_bool_and_integer_dtypes():
    cleaned = get_clean_full_confidence(
        {
            "token_has_frame": torch.tensor([True, False]),
            "token_asym_id": torch.tensor([1, 2], dtype=torch.int64),
            "scores": torch.tensor([1.234], dtype=torch.bfloat16),
        }
    )

    assert cleaned["token_has_frame"].dtype == np.bool_
    assert cleaned["token_asym_id"].dtype == np.int64
    assert cleaned["scores"].dtype == np.float32
    np.testing.assert_allclose(cleaned["scores"], [1.23])


@pytest.mark.parametrize(
    ("group_name", "seed", "message"),
    [
        ("../escape", 1, "safe path component"),
        ("group", "../../escape", "output seed must be an integer"),
        ("group", True, "output seed must be an integer, not a boolean"),
    ],
)
def test_dumper_rejects_unsafe_library_output_coordinates(
    tmp_path, group_name, seed, message
):
    dumper = DataDumper(str(tmp_path))

    with pytest.raises(ValueError, match=message):
        dumper._get_dump_dir(group_name, "sample", seed)


def test_structure_serialization_does_not_annotate_caller_atom_array(
    tmp_path, atom_array
):
    dumper = DataDumper(str(tmp_path))
    prediction = _minimal_prediction()
    prediction["full_data"] = [{"atom_plddt": torch.tensor([0.8765])}]
    dumper.dump(
        group_name="",
        pdb_id="job",
        pred_dict=prediction,
        atom_array=atom_array,
        entity_poly_type={},
        seed=1,
    )

    assert "b_factor" not in atom_array.get_annotation_categories()
    model = tmp_path / "job" / "models" / "seed-1_sample-0_model.cif"
    atoms = pdbx.CIFFile.read(model).block["atom_site"]
    np.testing.assert_allclose(atoms["B_iso_or_equiv"].as_array(float), [87.65])


def test_output_directories_keep_umask_permissions(tmp_path, atom_array):
    dumper = DataDumper(str(tmp_path), need_atom_confidence=True)

    previous_umask = os.umask(0o022)
    try:
        dumper.dump(
            pred_dict=_minimal_prediction(),
            group_name="",
            pdb_id="job",
            atom_array=atom_array,
            entity_poly_type={},
            seed=1,
        )
    finally:
        os.umask(previous_umask)

    for directory in ("models", "summary_confidences", "full_data"):
        mode = stat.S_IMODE((tmp_path / "job" / directory).stat().st_mode)
        assert mode == 0o755


def test_compressed_full_confidence_writes_npz_and_removes_stale_json(
    tmp_path, atom_array
):
    full_data_dir = tmp_path / "job" / "full_data"
    full_data_dir.mkdir(parents=True)
    stale_json = full_data_dir / "seed-7_sample-0_full_data.json"
    stale_json.write_text('{"stale": true}', encoding="utf-8")
    prediction = _minimal_prediction()
    prediction["full_data"] = [
        {
            "atom_plddt": torch.tensor([0.8765]),
            "token_pair_pae": np.array([[0.0, 0.254]]),
            "atom_coordinate": torch.ones(1, 3),
            "atom_is_polymer": torch.ones(1, dtype=torch.bool),
        }
    ]

    DataDumper(
        str(tmp_path),
        need_atom_confidence=True,
        compress_full_confidence=True,
    ).dump(
        group_name="",
        pdb_id="job",
        seed=7,
        pred_dict=prediction,
        atom_array=atom_array,
        entity_poly_type={"1": "polypeptide(L)"},
    )

    archive_path = full_data_dir / "seed-7_sample-0_full_data.npz"
    assert archive_path.is_file()
    assert not stale_json.exists()
    with np.load(archive_path, allow_pickle=False) as archive:
        assert set(archive.files) == {"atom_plddt", "token_pair_pae"}
        np.testing.assert_allclose(archive["atom_plddt"], [0.88])
        np.testing.assert_allclose(archive["token_pair_pae"], [[0.0, 0.25]])
        assert all(not archive[key].dtype.hasobject for key in archive.files)


def test_json_full_confidence_removes_stale_npz_after_success(tmp_path, atom_array):
    full_data_dir = tmp_path / "job" / "full_data"
    full_data_dir.mkdir(parents=True)
    stale_npz = full_data_dir / "seed-9_sample-0_full_data.npz"
    np.savez_compressed(stale_npz, stale=np.array([1]))

    DataDumper(
        str(tmp_path),
        need_atom_confidence=True,
        compress_full_confidence=False,
    ).dump(
        group_name="",
        pdb_id="job",
        seed=9,
        pred_dict={
            **_minimal_prediction(),
            "full_data": [{"atom_plddt": torch.tensor([0.5])}],
        },
        atom_array=atom_array,
        entity_poly_type={"1": "polypeptide(L)"},
    )

    assert json.loads(
        (full_data_dir / "seed-9_sample-0_full_data.json").read_text()
    ) == {"atom_plddt": [0.5]}
    assert not stale_npz.exists()


def test_disabling_full_confidence_removes_stale_formats(tmp_path, atom_array):
    full_data_dir = tmp_path / "job" / "full_data"
    full_data_dir.mkdir(parents=True)
    json_path = full_data_dir / "seed-4_sample-0_full_data.json"
    npz_path = full_data_dir / "seed-4_sample-0_full_data.npz"
    json_path.write_text('{"stale": true}', encoding="utf-8")
    np.savez_compressed(npz_path, stale=np.array([1]))

    DataDumper(str(tmp_path), need_atom_confidence=False).dump(
        group_name="",
        pdb_id="job",
        seed=4,
        pred_dict=_minimal_prediction(),
        atom_array=atom_array,
        entity_poly_type={"1": "polypeptide(L)"},
    )

    assert not json_path.exists()
    assert not npz_path.exists()


def test_failed_npz_publication_preserves_existing_json(
    tmp_path, atom_array, monkeypatch
):
    full_data_dir = tmp_path / "job" / "full_data"
    full_data_dir.mkdir(parents=True)
    stale_json = full_data_dir / "seed-5_sample-0_full_data.json"
    stale_json.write_text('{"still": "usable"}', encoding="utf-8")
    monkeypatch.setattr(
        np,
        "savez_compressed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        DataDumper(
            str(tmp_path),
            need_atom_confidence=True,
            compress_full_confidence=True,
        ).dump(
            group_name="",
            pdb_id="job",
            seed=5,
            pred_dict=_minimal_prediction(),
            atom_array=atom_array,
            entity_poly_type={"1": "polypeptide(L)"},
        )

    assert json.loads(stale_json.read_text()) == {"still": "usable"}
