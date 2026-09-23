"""Public publication defaults and confidence-format round trips."""
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from runner import batch_inference
from runner.dumper import DataDumper
from runner.fold_input import load_input_jobs, write_prepared_job
from runner.prediction_resume import seed_outputs_complete
from opendde.utils.text_io import read_text
from tests.test_dumper_reliability import atom_array, _minimal_prediction


def _job():
    return {"name": "job", "modelSeeds": [7], "sequences": [{"proteinChain": {
        "sequence": "A", "count": 1, "pairedMsa": ">q\nA\n",
        "unpairedMsa": ">q\nA\n>x\nW\n", "templates": [],
    }}]}


@pytest.mark.parametrize("compressed", [None, False, True])
def test_input_publication_roundtrip(tmp_path, compressed):
    source = Path(write_prepared_job(_job(), tmp_path / "old", compress_fold_input=True))
    job = load_input_jobs(str(source))[0][1]
    options = {} if compressed is None else {"compress_fold_input": compressed}
    published = Path(write_prepared_job(job, tmp_path / "new", **options))
    chain = json.loads(published.read_text())[0]["sequences"][0]["proteinChain"]
    assert "unpairedMsa" not in chain
    suffix = ".a3m.zst" if compressed else ".a3m"
    assert chain["unpairedMsaPath"] == "msas/job__A_unpairedmsa" + suffix
    assert read_text(published.parent / chain["unpairedMsaPath"]) == ">q\nA\n>x\nW\n"
    assert read_text(load_input_jobs(str(published))[0][1]["sequences"][0]["proteinChain"]["unpairedMsaPath"]) == ">q\nA\n>x\nW\n"


@pytest.mark.parametrize("write", [None, False, True])
def test_inference_only_cli_refreshes_snapshot_without_search(tmp_path, monkeypatch, write):
    source = tmp_path / "input.json"
    source.write_text(json.dumps([_job()]))
    output = tmp_path / "out"
    for relative in ("models/seed-7_sample-0_model.cif", "summary_confidences/seed-7_sample-0_summary_confidences.json", "full_data/seed-7_sample-0_full_data.json"):
        path = output / "job" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("already complete")
    def forbidden(*args, **kwargs):
        pytest.fail("completed inference must neither search nor load a model")
    monkeypatch.setattr(batch_inference, "get_default_runner", forbidden)
    monkeypatch.setattr(batch_inference, "msa_search", forbidden)
    args = ["-i", str(source), "-o", str(output), "-D", "false", "--sample", "1", "--skip", "true"]
    if write is not None:
        args += ["-J", str(write).lower()]
    result = CliRunner().invoke(batch_inference.predict, args)
    assert result.exit_code == 0, result.output
    published = output / "job/job_data.json"
    assert published.exists() is (write is not False)
    if published.exists():
        assert (output / "job/msas/job__A_unpairedmsa.a3m").read_text() == ">q\nA\n>x\nW\n"
        data = _job()
        data["sequences"][0]["proteinChain"]["unpairedMsa"] = ">q\nA\n>new\nG\n"
        source.write_text(json.dumps([data]))
        again = CliRunner().invoke(batch_inference.predict, args)
        assert again.exit_code == 0, again.output
        assert (output / "job/msas/job__A_unpairedmsa.a3m").read_text().endswith(">new\nG\n")


@pytest.mark.parametrize("compressed", [None, False, True])
def test_confidence_formats_preserve_shape_and_switch_cleanup(tmp_path, atom_array, compressed):
    prediction = _minimal_prediction()
    prediction["full_data"] = [{"atom_plddt": torch.tensor([0.5]), "token_pair_pae": np.array([[0.25]], dtype=np.float32), "token_asym_id": np.array([1], dtype=np.int64)}]
    embedded = tmp_path / "job/embeddings/seed-7_sample-0_embeddings.npz"
    embedded.parent.mkdir(parents=True)
    embedded.write_bytes(b"unrelated embeddings")
    for selection in (not bool(compressed), compressed):
        options = {} if selection is None else {"compress_full_confidence": selection}
        DataDumper(str(tmp_path), need_atom_confidence=True, **options).dump(
            group_name="", pdb_id="job", seed=7, pred_dict=prediction,
            atom_array=atom_array, entity_poly_type={"1": "polypeptide(L)"})
        full = tmp_path / ("job/full_data/seed-7_sample-0_full_data." + ("npz" if selection else "json"))
        assert full.is_file()
        assert not full.with_suffix(".json" if selection else ".npz").exists()
        if selection:
            with ZipFile(full) as archive:
                assert all(item.compress_type == ZIP_DEFLATED for item in archive.infolist())
            with np.load(full) as archive:
                actual = dict(archive)
                assert actual["token_asym_id"].dtype == np.dtype("int64")
        else:
            actual = json.loads(full.read_text())
        for key, expected in {"atom_plddt": [0.5], "token_pair_pae": [[0.25]], "token_asym_id": [1]}.items():
            np.testing.assert_array_equal(actual[key], expected)
        assert seed_outputs_complete(tmp_path, "job", 7, 1, need_atom_confidence=True, **options)
    full.unlink()
    assert not seed_outputs_complete(tmp_path, "job", 7, 1, need_atom_confidence=True, **options)
    assert embedded.read_bytes() == b"unrelated embeddings"
