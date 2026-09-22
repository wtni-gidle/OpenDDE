"""Explicit prepared conditions take precedence over automatic searches."""

from copy import deepcopy

import pytest

from opendde.data.inference.input_validation import validate_inference_jobs
from runner import msa_search


def job(chain):
    return {"name": "target", "sequences": [{"proteinChain": chain}]}


@pytest.mark.parametrize("value", [None, "", "hits.a3m"])
def test_public_input_rejects_legacy_templates_path(value):
    with pytest.raises(ValueError, match="templatesPath"):
        validate_inference_jobs(
            [job({"sequence": "ACD", "templatesPath": value, "templates": []})]
        )


@pytest.mark.parametrize("channel", ["pairedMsa", "unpairedMsa"])
def test_input_rejects_inline_and_path_for_the_same_channel(channel):
    with pytest.raises(ValueError, match=channel):
        validate_inference_jobs(
            [job({"sequence": "ACD", channel: "", channel + "Path": "some.a3m"})]
        )


@pytest.mark.parametrize("inline", ["", ">q\nACD\n"])
def test_inline_condition_does_not_trigger_msa_search(inline):
    assert (
        msa_search.need_msa_search(job({"sequence": "ACD", "unpairedMsa": inline}))
        is False
    )


def test_missing_explicit_msa_path_fails_instead_of_searching(tmp_path):
    with pytest.raises(FileNotFoundError, match="absent.a3m"):
        msa_search.need_msa_search(
            job({"sequence": "ACD", "unpairedMsaPath": str(tmp_path / "absent.a3m")})
        )


def test_search_for_other_chain_preserves_all_explicit_conditions(
    tmp_path, monkeypatch
):
    supplied = tmp_path / "deepmsa.a3m"
    supplied.write_text(">q\nACD\n>custom\nA-D\n")
    chain_a = {
        "sequence": "ACD",
        "count": 1,
        "unpairedMsaPath": str(supplied),
        "pairedMsa": "",
    }
    chain_b = {"sequence": "FGH", "count": 1}
    data = {
        "name": "target",
        "sequences": [{"proteinChain": chain_a}, {"proteinChain": chain_b}],
    }
    original_a = deepcopy(chain_a)

    def search(seqs, out_dir, mode=None):
        assert seqs == ["ACD", "FGH"]  # retain native full-complex search context
        result = []
        for index, sequence in enumerate(seqs):
            root = tmp_path / f"search-{index}"
            root.mkdir()
            (root / "pairing.a3m").write_text(f">q\n{sequence}\n")
            (root / "non_pairing.a3m").write_text(f">q\n{sequence}\n")
            result.append(str(root))
        return result

    monkeypatch.setattr(msa_search, "msa_search", search)
    msa_search.update_seq_msa(data, str(tmp_path / "scratch"))
    assert chain_a == original_a
    assert chain_b["pairedMsaPath"].endswith("search-1/pairing.a3m")
    assert supplied.read_text() == ">q\nACD\n>custom\nA-D\n"


@pytest.mark.parametrize("inline", ["", ">q\nACG\n"])
def test_rna_inline_is_not_replaced_by_search(tmp_path, monkeypatch, inline):
    from runner import rna_msa_search

    def unexpected(**kwargs):
        pytest.fail("explicit RNA MSA must not trigger search")

    monkeypatch.setattr(rna_msa_search, "run_rna_msa_search", unexpected)
    chain = {"sequence": "ACG", "unpairedMsa": inline}
    data = [{"name": "rna", "sequences": [{"rnaSequence": chain}]}]
    assert not rna_msa_search.update_rna_msa_info(data, str(tmp_path))
    assert chain == {"sequence": "ACG", "unpairedMsa": inline}


def test_prepared_inline_resources_and_failed_update_preserve_old_bundle(tmp_path):
    import json
    from pathlib import Path
    from runner.fold_input import write_prepared_job
    from opendde.utils.text_io import read_text

    original = job({"sequence": "ACD", "pairedMsa": "", "unpairedMsa": ">q\nACD\n"})
    path = Path(write_prepared_job(original, tmp_path))
    chain = json.loads(path.read_text())[0]["sequences"][0]["proteinChain"]
    assert "unpairedMsa" not in chain
    assert chain["unpairedMsaPath"] == "msas/target__A_unpairedmsa.a3m.zst"
    assert read_text(path.parent / chain["pairedMsaPath"]) == ""
    before = {p: p.read_bytes() for p in path.parent.rglob("*") if p.is_file()}
    changed = job(
        {
            "sequence": "ACD",
            "pairedMsa": ">q\nACD\n",
            "unpairedMsaPath": str(tmp_path / "missing.a3m"),
        }
    )
    with pytest.raises(FileNotFoundError):
        write_prepared_job(changed, tmp_path)
    assert {p: p.read_bytes() for p in path.parent.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("predict", [False, True])
def test_private_data_bundle_lives_through_prediction_only(
    tmp_path, monkeypatch, predict
):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from runner import batch_inference
    from opendde.utils.text_io import read_text

    source = tmp_path / "input.json"
    source.write_text(
        json.dumps([job({"sequence": "ACD", "unpairedMsa": ">q\nACD\n"})])
    )
    output = tmp_path / "public"
    scratch_root = tmp_path / "slurm"
    scratch_root.mkdir()
    monkeypatch.setenv("SLURM_TMPDIR", str(scratch_root))
    seen = []
    closed = []

    def infer(runner, configs):
        path = Path(configs["input_json_path"])
        assert path.is_relative_to(scratch_root)
        chain = json.loads(path.read_text())[0]["sequences"][0]["proteinChain"]
        assert read_text(path.parent / chain["unpairedMsaPath"]) == ">q\nACD\n"
        seen.append(path)

    monkeypatch.setattr(
        batch_inference,
        "get_default_runner",
        lambda **kwargs: SimpleNamespace(configs={}, close=lambda: closed.append(True)),
    )
    monkeypatch.setattr(batch_inference, "infer_predict", infer)
    paths = batch_inference.run_prediction_workflow(
        str(source),
        str(output),
        run_data_pipeline=True,
        run_inference=predict,
        write_input_json=False,
        use_msa=False,
        use_template=False,
        use_rna_msa=False,
    )
    assert paths == [str(source)]
    assert not output.exists()
    assert len(seen) == int(predict)
    assert closed == ([True] if predict else [])
    assert list(scratch_root.iterdir()) == []


def test_inference_only_can_publish_current_inline_input(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from runner import batch_inference
    from opendde.utils.text_io import read_text

    source = tmp_path / "input.json"
    source.write_text(
        json.dumps([job({"sequence": "ACD", "unpairedMsa": ">q\nACD\n"})])
    )
    monkeypatch.setattr(
        msa_search, "msa_search", lambda *a, **k: pytest.fail("no search")
    )
    monkeypatch.setattr(
        batch_inference,
        "get_default_runner",
        lambda **kwargs: SimpleNamespace(configs={}, close=lambda: None),
    )
    monkeypatch.setattr(batch_inference, "infer_predict", lambda *args: None)
    paths = batch_inference.run_prediction_workflow(
        str(source),
        str(tmp_path / "out"),
        run_data_pipeline=False,
        write_input_json=True,
    )
    path = Path(paths[0])
    assert path.name == "target_data.json"
    chain = json.loads(path.read_text())[0]["sequences"][0]["proteinChain"]
    assert read_text(path.parent / chain["unpairedMsaPath"]) == ">q\nACD\n"


def test_publication_error_rolls_back_resources_and_json(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    from runner.fold_input import write_prepared_job

    data = job({"sequence": "ACD", "unpairedMsa": ">q\nACD\n"})
    path = Path(write_prepared_job(data, tmp_path))
    before = {p: p.read_bytes() for p in path.parent.rglob("*") if p.is_file()}
    real_replace = os.replace

    def replace(source, destination):
        if Path(destination) == path:
            raise OSError("publication failed")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", replace)
    data["sequences"][0]["proteinChain"]["unpairedMsa"] = ">q\nACD\n>x\nA-D\n"
    with pytest.raises(OSError, match="publication failed"):
        write_prepared_job(data, tmp_path)
    assert {p: p.read_bytes() for p in path.parent.rglob("*") if p.is_file()} == before


def test_cli_write_switch_can_suppress_public_data_output(tmp_path, monkeypatch):
    import json
    from click.testing import CliRunner
    from runner.batch_inference import predict

    source = tmp_path / "input.json"
    source.write_text(json.dumps([job({"sequence": "ACD", "unpairedMsa": ""})]))
    output = tmp_path / "out"
    result = CliRunner().invoke(
        predict,
        [
            "-i",
            str(source),
            "-o",
            str(output),
            "-D",
            "true",
            "-P",
            "false",
            "-J",
            "false",
            "--use_template",
            "false",
        ],
    )
    assert result.exit_code == 0, result.output
    assert not output.exists()


def test_mt_produces_explicit_prepared_json_not_legacy_hits(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from runner.batch_inference import msatemplate

    source = tmp_path / "input.json"
    source.write_text(
        json.dumps(
            [
                job(
                    {
                        "sequence": "ACD",
                        "pairedMsa": "",
                        "unpairedMsa": ">q\nACD\n",
                        "templates": [],
                    }
                )
            ]
        )
    )
    monkeypatch.setattr(
        msa_search, "msa_search", lambda *a, **k: pytest.fail("supplied MSA")
    )
    paths = msatemplate.callback(
        str(source), str(tmp_path / "out"), None, None, None, None
    )
    assert isinstance(paths, list)
    prepared = json.loads(Path(paths[0]).read_text())[0]["sequences"][0]["proteinChain"]
    assert prepared["templates"] == []
    assert prepared["unpairedMsaPath"].startswith("msas/")


def test_data_rejects_bad_cif_without_publishing(tmp_path):
    import json
    from runner.fold_input import prepare_input_jobs

    source = tmp_path / "input.json"
    source.write_text(
        json.dumps(
            [
                job(
                    {
                        "sequence": "ACD",
                        "templates": [
                            {
                                "mmcif": "data_broken\n",
                                "queryIndices": [0],
                                "templateIndices": [0],
                            }
                        ],
                    }
                )
            ]
        )
    )
    with pytest.raises(ValueError, match="template|mmCIF|CIF"):
        prepare_input_jobs(str(source), str(tmp_path / "out"), use_msa=False)
    assert not (tmp_path / "out").exists()


def test_native_tool_scratch_uses_slurm_directory(tmp_path, monkeypatch):
    from pathlib import Path
    from opendde.data.tools.common import tmpdir_manager

    slurm = tmp_path / "slurm"
    slurm.mkdir()
    monkeypatch.setenv("SLURM_TMPDIR", str(slurm))
    with tmpdir_manager() as directory:
        assert Path(directory).parent == slurm
        assert Path(directory).is_dir()
    assert list(slurm.iterdir()) == []


def test_legacy_msa_does_not_fill_other_channel_when_modern_condition_is_supplied(
    tmp_path,
):
    (tmp_path / "pairing.a3m").write_text(">q\nACD\n")
    chain = {
        "sequence": "ACD",
        "unpairedMsa": "",
        "msa": {"precomputed_msa_dir": str(tmp_path)},
    }
    converted, _ = msa_search.convert_one_json_dict(job(chain))
    assert "pairedMsaPath" not in converted["sequences"][0]["proteinChain"]


def test_publish_without_search_materializes_native_msa_directory(
    tmp_path, monkeypatch
):
    import json
    from pathlib import Path
    from runner.fold_input import prepare_input_jobs
    from opendde.utils.text_io import read_text

    native = tmp_path / "native"
    native.mkdir()
    (native / "pairing.a3m").write_text(">q\nACD\n")
    (native / "non_pairing.a3m").write_text(">q\nACD\n>hit\nA-D\n")
    source = tmp_path / "input.json"
    source.write_text(
        json.dumps([job({"sequence": "ACD", "msa": {"precomputed_msa_dir": "native"}})])
    )
    monkeypatch.setattr(
        msa_search, "msa_search", lambda *a, **k: pytest.fail("no search")
    )
    [path] = prepare_input_jobs(str(source), str(tmp_path / "out"), use_msa=False)
    chain = json.loads(Path(path).read_text())[0]["sequences"][0]["proteinChain"]
    assert "msa" not in chain
    assert chain["pairedMsaPath"].startswith("msas/")
    assert (
        read_text(Path(path).parent / chain["unpairedMsaPath"])
        == ">q\nACD\n>hit\nA-D\n"
    )


def test_publication_allows_trusted_root_alias_but_not_resource_symlinks(tmp_path):
    from pathlib import Path
    from runner.fold_input import write_prepared_job

    root = tmp_path / "real"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    data = job({"sequence": "ACD", "unpairedMsa": ">q\nACD\n"})
    path = Path(write_prepared_job(data, alias))
    assert path == alias / "target" / "target_data.json"
    assert path.is_file()
    resource = root / "target" / "msas" / "target__A_unpairedmsa.a3m.zst"
    resource.unlink()
    protected = tmp_path / "protected"
    protected.write_bytes(b"unchanged")
    resource.symlink_to(protected)
    with pytest.raises(ValueError, match="symlink"):
        write_prepared_job(data, alias)
    assert protected.read_bytes() == b"unchanged"
