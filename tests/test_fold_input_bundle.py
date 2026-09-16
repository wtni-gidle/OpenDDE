# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import json
from pathlib import Path

from runner.fold_input import load_input_jobs, write_prepared_job


def _write_fixture_input(
    tmp_path: Path,
) -> tuple[Path, dict[str, str], dict[str, str]]:
    source_dir = tmp_path / "source"
    assets_dir = source_dir / "assets"
    assets_dir.mkdir(parents=True)
    contents = {
        "paired": ">paired\nACD\n",
        "unpaired": ">unpaired\nACD\n",
        "template": "data_template\n#\n",
        "template_hits": ">hits\nACD\n",
        "rna": ">rna\nAUG\n",
    }
    paths = {
        "paired": assets_dir / "paired.a3m",
        "unpaired": assets_dir / "unpaired.a3m",
        "template": assets_dir / "template.cif",
        "template_hits": assets_dir / "template_hits.hhr",
        "rna": assets_dir / "rna.a3m",
    }
    for name, path in paths.items():
        path.write_text(contents[name])

    source_json = source_dir / "arbitrarily_named_input.json"
    source_json.write_text(
        json.dumps(
            [
                {
                    "name": "target",
                    "sequences": [
                        {
                            "proteinChain": {
                                "sequence": "ACD",
                                "id": ["A", "B"],
                                "pairedMsaPath": "assets/paired.a3m",
                                "unpairedMsaPath": "assets/unpaired.a3m",
                                "templatesPath": "assets/template_hits.hhr",
                                "templates": [
                                    {
                                        "mmcifPath": "assets/template.cif",
                                        "queryIndices": [0, 1, 2],
                                        "templateIndices": [0, 1, 2],
                                    }
                                ],
                            }
                        },
                        {
                            "rnaSequence": {
                                "sequence": "AUG",
                                "id": ["R"],
                                "unpairedMsaPath": "assets/rna.a3m",
                            }
                        },
                    ],
                    "unknown": "preserved",
                }
            ]
        )
    )
    return source_json, {name: str(path) for name, path in paths.items()}, contents


def test_load_input_jobs_resolves_resource_paths_from_json_directory(
    tmp_path: Path, monkeypatch
):
    """Changing the process CWD must not change how source resource paths resolve."""
    source_json, resource_paths, _ = _write_fixture_input(tmp_path)
    monkeypatch.chdir(tmp_path)

    loaded_path, loaded = load_input_jobs(str(source_json))[0]
    protein = loaded["sequences"][0]["proteinChain"]
    rna = loaded["sequences"][1]["rnaSequence"]

    assert loaded_path == source_json
    assert protein["pairedMsaPath"] == resource_paths["paired"]
    assert protein["unpairedMsaPath"] == resource_paths["unpaired"]
    assert protein["templatesPath"] == resource_paths["template_hits"]
    assert protein["templates"][0]["mmcifPath"] == resource_paths["template"]
    assert rna["unpairedMsaPath"] == resource_paths["rna"]


def test_write_prepared_job_makes_portable_target_bundle(tmp_path: Path):
    """Omitting a copied resource or leaving its source path breaks a portable bundle."""
    source_json, _, contents = _write_fixture_input(tmp_path)
    _, job = load_input_jobs(str(source_json))[0]

    prepared = write_prepared_job(job, tmp_path / "out")
    job_dir = tmp_path / "out" / "target"
    loaded = json.loads(Path(prepared).read_text())

    assert prepared == str(job_dir / "target_data.json")
    assert (job_dir / "msas" / "target__A_pairedmsa.a3m").read_text() == contents[
        "paired"
    ]
    assert (job_dir / "msas" / "target__A_unpairedmsa.a3m").read_text() == contents[
        "unpaired"
    ]
    assert (job_dir / "msas" / "target__A_template_0.cif").read_text() == contents[
        "template"
    ]
    assert loaded[0]["unknown"] == "preserved"
    assert loaded[0]["sequences"][0]["proteinChain"]["pairedMsaPath"] == (
        "msas/target__A_pairedmsa.a3m"
    )
    assert (job_dir / "msas" / "target__R_unpairedmsa.a3m").read_text() == contents[
        "rna"
    ]
    assert loaded[0]["sequences"][1]["rnaSequence"]["unpairedMsaPath"] == (
        "msas/target__R_unpairedmsa.a3m"
    )


def test_write_prepared_job_uses_name_based_directories_for_multiple_jobs(
    tmp_path: Path,
):
    """Writing a second job must not overwrite the first job's bundle."""
    source_json, _, _ = _write_fixture_input(tmp_path)
    _, first = load_input_jobs(str(source_json))[0]
    second = json.loads(json.dumps(first))
    second["name"] = "second"

    first_prepared = write_prepared_job(first, tmp_path / "out")
    second_prepared = write_prepared_job(second, tmp_path / "out")

    assert first_prepared == str(tmp_path / "out" / "target" / "target_data.json")
    assert second_prepared == str(tmp_path / "out" / "second" / "second_data.json")


def test_write_prepared_job_assigns_fallback_labels_without_overwriting_msa(
    tmp_path: Path,
):
    """A fallback RNA label must not overwrite an earlier explicit protein label."""
    protein_msa = tmp_path / "protein.a3m"
    rna_msa = tmp_path / "rna.a3m"
    protein_msa.write_text(">protein\nACD\n")
    rna_msa.write_text(">rna\nAUG\n")
    job = {
        "name": "mixed",
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "ACD",
                    "id": ["B"],
                    "unpairedMsaPath": str(protein_msa),
                }
            },
            {"rnaSequence": {"sequence": "AUG", "unpairedMsaPath": str(rna_msa)}},
        ],
    }

    prepared = write_prepared_job(job, tmp_path / "out")
    job_dir = tmp_path / "out" / "mixed"
    loaded = json.loads(Path(prepared).read_text())

    assert (
        job_dir / "msas" / "mixed__B_unpairedmsa.a3m"
    ).read_text() == ">protein\nACD\n"
    assert (job_dir / "msas" / "mixed__A_unpairedmsa.a3m").read_text() == ">rna\nAUG\n"
    assert loaded[0]["sequences"][0]["proteinChain"]["unpairedMsaPath"] == (
        "msas/mixed__B_unpairedmsa.a3m"
    )
    assert loaded[0]["sequences"][1]["rnaSequence"]["unpairedMsaPath"] == (
        "msas/mixed__A_unpairedmsa.a3m"
    )


def test_file_ligand_is_json_relative_and_stays_external_in_prepared_bundle(
    tmp_path, monkeypatch
):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    ligand_file = source_dir / "ligand.sdf"
    ligand_file.write_text("external ligand")
    source_json = source_dir / "input.json"
    source_json.write_text(
        json.dumps(
            [
                {
                    "name": "ligand_job",
                    "sequences": [
                        {"ligand": {"ligand": "FILE_ligand.sdf", "count": 1}},
                        {"ligand": {"ligand": "CCD_ATP", "count": 1}},
                    ],
                }
            ]
        )
    )
    monkeypatch.chdir(tmp_path)

    _, job = load_input_jobs(str(source_json))[0]
    expected = f"FILE_{ligand_file}"
    assert job["sequences"][0]["ligand"]["ligand"] == expected
    assert job["sequences"][1]["ligand"]["ligand"] == "CCD_ATP"
    prepared = Path(write_prepared_job(job, tmp_path / "output"))
    assert (
        json.loads(prepared.read_text())[0]["sequences"][0]["ligand"]["ligand"]
        == expected
    )
    assert list((prepared.parent / "msas").iterdir()) == []

    moved_dir = tmp_path / "moved"
    prepared.parent.rename(moved_dir)
    _, reloaded = load_input_jobs(str(moved_dir / prepared.name))[0]
    assert reloaded["sequences"][0]["ligand"]["ligand"] == expected
    assert ligand_file.read_text() == "external ligand"


def test_prepared_file_ligand_resolves_against_prepared_json(tmp_path, monkeypatch):
    job_dir = tmp_path / "prepared"
    job_dir.mkdir()
    prepared = job_dir / "job_data.json"
    prepared.write_text(
        json.dumps(
            [
                {
                    "name": "job",
                    "sequences": [{"ligand": {"ligand": "FILE_local.sdf", "count": 1}}],
                }
            ]
        )
    )
    monkeypatch.chdir(tmp_path)

    _, job = load_input_jobs(str(prepared))[0]
    assert job["sequences"][0]["ligand"]["ligand"] == f"FILE_{job_dir / 'local.sdf'}"
