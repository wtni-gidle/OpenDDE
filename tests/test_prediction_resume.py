# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Tests for lightweight prediction resume behavior."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner
import pytest
import torch
import numpy as np

from runner import batch_inference, inference


def _prediction_resume():
    """Import lazily so a missing implementation is a RED test, not collection."""
    return importlib.import_module("runner.prediction_resume")


def seed_outputs_complete(*args, **kwargs):
    return _prediction_resume().seed_outputs_complete(*args, **kwargs)


def incomplete_job_seed_schedule(*args, **kwargs):
    return _prediction_resume().incomplete_job_seed_schedule(*args, **kwargs)


def _sample_paths(root: Path, job: str, seed: int, sample: int) -> dict[str, Path]:
    prefix = f"seed-{seed}_sample-{sample}"
    job_dir = root / job
    return {
        "model": job_dir / "models" / f"{prefix}_model.cif",
        "summary": (
            job_dir / "summary_confidences" / f"{prefix}_summary_confidences.json"
        ),
        "full": job_dir / "full_data" / f"{prefix}_full_data.json",
        "full_npz": job_dir / "full_data" / f"{prefix}_full_data.npz",
    }


def _write_complete_seed(
    root: Path,
    job: str,
    seed: int,
    samples: int,
    *,
    full_data: bool = True,
) -> None:
    for sample in range(samples):
        paths = _sample_paths(root, job, seed, sample)
        for path in (paths["model"], paths["summary"], paths["full"]):
            path.parent.mkdir(parents=True, exist_ok=True)
        paths["model"].write_text("data_model\n#\n", encoding="utf-8")
        paths["summary"].write_text(
            json.dumps({"ranking_score": 0.8, "sample": sample}),
            encoding="utf-8",
        )
        if full_data:
            paths["full"].write_text(
                json.dumps({"atom_plddt": [0.8], "sample": sample}),
                encoding="utf-8",
            )


def _write_complete_npz_seed(root: Path, job: str, seed: int, samples: int) -> None:
    for sample in range(samples):
        paths = _sample_paths(root, job, seed, sample)
        for path in (paths["model"], paths["summary"], paths["full_npz"]):
            path.parent.mkdir(parents=True, exist_ok=True)
        paths["model"].write_text("data_model\n#\n", encoding="utf-8")
        paths["summary"].write_text('{"ranking_score": 0.8}', encoding="utf-8")
        np.savez_compressed(paths["full_npz"], atom_plddt=np.array([0.8]))


def _write_input(path: Path, jobs: list[dict]) -> Path:
    path.write_text(json.dumps(jobs), encoding="utf-8")
    return path


def test_seed_outputs_require_each_canonical_sample_and_optional_full_data(tmp_path):
    _write_complete_seed(tmp_path, "job", 7, 2, full_data=False)

    assert seed_outputs_complete(tmp_path, "job", 7, 2, need_atom_confidence=False)
    assert not seed_outputs_complete(tmp_path, "job", 7, 3, need_atom_confidence=False)
    assert not seed_outputs_complete(tmp_path, "job", 7, 2, need_atom_confidence=True)


def test_seed_outputs_require_requested_full_confidence_format(tmp_path):
    _write_complete_npz_seed(tmp_path, "npz_job", 7, 1)
    _write_complete_seed(tmp_path, "json_job", 7, 1)

    assert seed_outputs_complete(
        tmp_path,
        "npz_job",
        7,
        1,
        need_atom_confidence=True,
        compress_full_confidence=True,
    )
    assert not seed_outputs_complete(
        tmp_path,
        "npz_job",
        7,
        1,
        need_atom_confidence=True,
        compress_full_confidence=False,
    )
    assert seed_outputs_complete(
        tmp_path,
        "json_job",
        7,
        1,
        need_atom_confidence=True,
        compress_full_confidence=False,
    )
    assert not seed_outputs_complete(
        tmp_path,
        "json_job",
        7,
        1,
        need_atom_confidence=True,
        compress_full_confidence=True,
    )


@pytest.mark.parametrize(
    ("target", "contents", "complete"),
    [
        ("model", b"", False),
        ("summary", b"", False),
        ("summary", b"not-json", True),
        ("summary", b"{}", True),
        ("full", b"", False),
        ("full", b"[]", True),
        ("full", b"{}", True),
    ],
)
def test_only_zero_byte_required_output_is_incomplete(tmp_path, target, contents, complete):
    _write_complete_seed(tmp_path, "job", 8, 1)
    _sample_paths(tmp_path, "job", 8, 0)[target].write_bytes(contents)

    assert seed_outputs_complete(tmp_path, "job", 8, 1, need_atom_confidence=True,
                                 compress_full_confidence=False) is complete


@pytest.mark.parametrize("compress", [False, True])
def test_nonempty_outputs_are_never_opened(tmp_path, monkeypatch, compress):
    for sample in range(2):
        for path in _sample_paths(tmp_path, "job", 7, sample).values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"garbage")
    def fail_open(*_args, **_kwargs):
        pytest.fail("resume opened an output")
    monkeypatch.setattr(Path, "open", fail_open)
    monkeypatch.setattr("builtins.open", fail_open)
    monkeypatch.setattr(np, "load", fail_open)
    assert seed_outputs_complete(tmp_path, "job", 7, 2, need_atom_confidence=True,
                                 compress_full_confidence=compress)


@pytest.mark.parametrize("target", ["model", "summary", "full", "full_npz"])
@pytest.mark.parametrize("damage", ["missing", "empty", "directory", "escape"])
def test_damage_selects_only_affected_seed(tmp_path, target, damage):
    compress = target == "full_npz"
    for seed in (7, 8):
        writer = _write_complete_npz_seed if compress else _write_complete_seed
        writer(tmp_path, "job", seed, 2)
    path = _sample_paths(tmp_path, "job", 8, 1)[target]
    path.unlink()
    if damage == "empty":
        path.touch()
    elif damage == "directory":
        path.mkdir()
    elif damage == "escape":
        outside = tmp_path / "outside"
        outside.write_bytes(b"nonempty")
        path.symlink_to(outside)
    assert incomplete_job_seed_schedule(
        tmp_path, [{"name": "job", "sequences": [{"proteinChain": {"sequence": "CHANGED"}}]}],
        [[7, 8]], 2, need_atom_confidence=True,
        compress_full_confidence=compress) == [[8]]


def test_incomplete_schedule_preserves_per_job_seed_order(tmp_path):
    jobs = [{"name": "first"}, {"name": "second"}]
    schedule = [[9, 7, 5], [4, 3]]
    _write_complete_seed(tmp_path, "first", 7, 2)
    _write_complete_seed(tmp_path, "second", 4, 2)

    assert incomplete_job_seed_schedule(
        tmp_path,
        jobs,
        schedule,
        2,
        need_atom_confidence=True,
        compress_full_confidence=False,
    ) == [[9, 5], [3]]


def test_all_complete_skips_before_runner_initialization(tmp_path, monkeypatch):
    source = _write_input(
        tmp_path / "job_data.json",
        [{"name": "job", "modelSeeds": [7, 8], "sequences": []}],
    )
    output = tmp_path / "output"
    _write_complete_seed(output, "job", 7, 2, full_data=False)
    _write_complete_seed(output, "job", 8, 2, full_data=False)

    monkeypatch.setattr(
        batch_inference,
        "get_default_runner",
        lambda **_kwargs: pytest.fail("complete outputs must not initialize runner"),
    )

    assert batch_inference.run_prediction_workflow(
        str(source),
        str(output),
        run_data_pipeline=False,
        run_inference=True,
        n_sample=2,
        need_atom_confidence=False,
        skip=True,
        n_step=99,
        n_cycle=9,
        write_input_json=False,
    ) == [str(source)]


@pytest.mark.parametrize(
    "workflow_args",
    [
        {"run_data_pipeline": False, "run_inference": True, "skip": False},
        {"run_data_pipeline": True, "run_inference": False, "skip": True},
        {
            "run_data_pipeline": False,
            "run_inference": True,
            "skip": True,
            "foldcp_mode": "distributed",
            "write_input_json": False,
        },
    ],
)
def test_workflows_without_local_resume_precheck_do_not_inspect_outputs(
    tmp_path, monkeypatch, workflow_args
):
    source = _write_input(
        tmp_path / "input.json",
        [{"name": "job", "modelSeeds": [7], "sequences": []}],
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        batch_inference,
        "_all_requested_outputs_complete",
        lambda *_args, **_kwargs: pytest.fail("prediction outputs were inspected"),
    )
    monkeypatch.setattr(
        batch_inference,
        "prepare_input_jobs",
        lambda *_args, **_kwargs: [str(source)],
    )
    monkeypatch.setattr(
        batch_inference,
        "get_default_runner",
        lambda **_kwargs: SimpleNamespace(configs={}, close=lambda: None),
    )
    monkeypatch.setattr(batch_inference, "infer_predict", lambda *_args: None)

    batch_inference.run_prediction_workflow(str(source), str(output), **workflow_args)


def test_distributed_resume_schedule_is_checked_only_on_rank0(tmp_path, monkeypatch):
    control_group = object()
    broadcasts = []
    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(
        inference,
        "incomplete_job_seed_schedule",
        lambda *_args, **_kwargs: pytest.fail(
            "nonzero rank must not inspect prediction outputs"
        ),
    )

    def broadcast(payload, *, src, group):
        broadcasts.append((src, group))
        payload[0] = (True, [[8]])

    monkeypatch.setattr(inference, "_broadcast_object_list", broadcast)

    assert inference._incomplete_job_seed_schedule_synchronized(
        tmp_path,
        [{"name": "job"}],
        [[7, 8]],
        1,
        need_atom_confidence=False,
        world_control_group=control_group,
    ) == [[8]]
    assert broadcasts == [(0, control_group)]


def test_distributed_resume_schedule_broadcasts_rank0_failure(tmp_path, monkeypatch):
    control_group = object()
    broadcasts = []
    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        inference,
        "incomplete_job_seed_schedule",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("storage offline")),
    )

    def broadcast(payload, *, src, group):
        broadcasts.append((payload[0], src, group))

    monkeypatch.setattr(inference, "_broadcast_object_list", broadcast)

    with pytest.raises(ValueError, match="Invalid prediction resume schedule"):
        inference._incomplete_job_seed_schedule_synchronized(
            tmp_path,
            [{"name": "job"}],
            [[7]],
            1,
            need_atom_confidence=False,
            world_control_group=control_group,
        )

    assert broadcasts == [((False, "OSError: storage offline"), 0, control_group)]


class _AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def test_direct_infer_predict_skips_complete_outputs_before_dataloader(
    tmp_path, monkeypatch
):
    source = _write_input(
        tmp_path / "job_data.json",
        [{"name": "job", "modelSeeds": [7], "sequences": []}],
    )
    output = tmp_path / "output"
    _write_complete_seed(output, "job", 7, 2, full_data=False)
    configs = _AttrDict(
        input_json_path=str(source),
        seeds=[],
        dump_dir=str(output),
        sample_diffusion=SimpleNamespace(N_sample=2),
        model=SimpleNamespace(N_model_seed=1),
        need_atom_confidence=False,
        skip=True,
    )
    runner = SimpleNamespace(
        foldcp_config=SimpleNamespace(enabled=False),
        foldcp_world_control_group=None,
        foldcp_control_group=None,
        foldcp_cp_rank=0,
        error_dir=str(tmp_path / "errors"),
    )
    monkeypatch.setattr(
        inference,
        "_create_inference_dataloader_synchronized",
        lambda *_args, **_kwargs: pytest.fail("complete outputs must not load data"),
    )

    inference.infer_predict(runner, configs)


def test_direct_infer_predict_reruns_an_incomplete_seed(tmp_path, monkeypatch):
    source = _write_input(
        tmp_path / "job_data.json",
        [{"name": "job", "modelSeeds": [7, 8], "sequences": []}],
    )
    output = tmp_path / "output"
    _write_complete_seed(output, "job", 7, 1, full_data=False)
    configs = _AttrDict(
        input_json_path=str(source),
        seeds=[],
        dump_dir=str(output),
        sample_diffusion=SimpleNamespace(N_sample=1),
        model=SimpleNamespace(N_model_seed=1),
        need_atom_confidence=False,
        skip=True,
    )
    runner = SimpleNamespace(
        foldcp_config=SimpleNamespace(enabled=False),
        foldcp_world_control_group=None,
        foldcp_control_group=None,
        foldcp_cp_rank=0,
        error_dir=str(tmp_path / "errors"),
    )
    reached_dataloader = []

    def stop_at_dataloader(*_args, **_kwargs):
        reached_dataloader.append(True)
        raise RuntimeError("stop after resume selection")

    monkeypatch.setattr(
        inference,
        "_create_inference_dataloader_synchronized",
        stop_at_dataloader,
    )

    with pytest.raises(RuntimeError, match="stop after resume selection"):
        inference.infer_predict(runner, configs)

    assert reached_dataloader == [True]


@pytest.mark.parametrize("damage", ["missing", "empty"])
def test_partial_sample_reruns_entire_seed_and_preserves_completed_seed(
    tmp_path, monkeypatch, damage
):
    source = _write_input(tmp_path / "input.json", [
        {"name": "job", "modelSeeds": [7, 8], "sequences": []}
    ])
    output = tmp_path / "output"
    for seed in (7, 8):
        _write_complete_seed(output, "job", seed, 4, full_data=False)
    path = _sample_paths(output, "job", 8, 3)["summary"]
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"")
    preserved = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for sample in range(4)
        for key, path in _sample_paths(output, "job", 7, sample).items()
        if key in ("model", "summary")
    }
    configs = _AttrDict(
        input_json_path=str(source), seeds=[], dump_dir=str(output),
        sample_diffusion=SimpleNamespace(N_sample=2),
        model=SimpleNamespace(N_model_seed=2),
        skip_amp=SimpleNamespace(), deterministic=False,
        need_atom_confidence=False, skip=True,
    )
    data = {"sample_name": "job", "sample_index": 0,
            "N_asym": 1, "N_token": 1, "N_atom": 1, "N_msa": 1,
            "input_feature_dict": {}, "entity_poly_type": {}}
    dataset = [(data, None, "")]
    sampler = inference.InferenceJobSampler(dataset, num_replicas=1, rank=0)
    loader = torch.utils.data.DataLoader(dataset, sampler=sampler,
                                        collate_fn=lambda batch: batch)
    monkeypatch.setattr(inference, "_create_inference_dataloader_synchronized",
                        lambda *_args: loader)
    predicted = []
    dumped = []

    def predict(batch):
        seed = int(batch["input_feature_dict"]["inference_seed"])
        predicted.append((seed, configs.sample_diffusion.N_sample,
                          configs.model.N_model_seed))
        return {"samples": [0, 1, 2, 3]}

    def dump(*, seed, pred_dict, **_kwargs):
        dumped.append((seed, pred_dict["samples"]))
        _write_complete_seed(output, "job", seed, 4, full_data=False)

    runner = SimpleNamespace(
        foldcp_config=SimpleNamespace(enabled=False),
        foldcp_world_control_group=None, foldcp_control_group=None, foldcp_cp_rank=0,
        error_dir=str(tmp_path / "errors"), device=torch.device("cpu"),
        update_model_configs=lambda _configs: None, predict=predict,
        dumper=SimpleNamespace(dump=dump),
    )
    inference.infer_predict(runner, configs)
    assert predicted == [(8, 2, 2)]
    assert dumped == [(8, [0, 1, 2, 3])]
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == before
               for path, before in preserved.items())


@pytest.mark.parametrize(
    ("extra_args", "expected_skip"),
    [
        ([], False),
        (["--skip", "true"], True),
    ],
)
def test_cli_forwards_skip(tmp_path, monkeypatch, extra_args, expected_skip):
    captured = {}
    monkeypatch.setattr(batch_inference, "init_logging", lambda: None)
    monkeypatch.setattr(
        batch_inference,
        "run_prediction_workflow",
        lambda *_args, **kwargs: captured.update(kwargs) or [],
    )

    result = CliRunner().invoke(
        batch_inference.predict,
        [
            "--input",
            str(tmp_path / "input.json"),
            "--run_data_pipeline",
            "false",
            "--run_inference",
            "true",
            *extra_args,
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["skip"] is expected_skip
    assert "write_now" not in captured


def test_cli_rejects_removed_write_now_option(tmp_path):
    (tmp_path / "input.json").write_text("[]")
    result = CliRunner().invoke(
        batch_inference.predict,
        [
            "--input",
            str(tmp_path / "input.json"),
            "--write_now",
            "false",
        ],
    )

    assert result.exit_code == 2
    assert "No such option '--write_now'" in result.output


def test_cli_forwards_compression_defaults_and_overrides(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(batch_inference, "init_logging", lambda: None)
    monkeypatch.setattr(
        batch_inference,
        "run_prediction_workflow",
        lambda *_args, **kwargs: captured.update(kwargs) or [],
    )

    default_result = CliRunner().invoke(
        batch_inference.predict,
        ["--input", str(tmp_path / "input.json"), "--run_data_pipeline", "false"],
    )
    assert default_result.exit_code == 0, default_result.output
    assert captured["compress_fold_input"] is False
    assert captured["compress_full_confidence"] is False
    captured.clear()

    for compression_value, expected in (("false", False), ("true", True)):
        captured.clear()
        result = CliRunner().invoke(
            batch_inference.predict,
            [
                "--input",
                str(tmp_path / "input.json"),
                "--run_data_pipeline",
                "false",
                "--compress_fold_input",
                "false",
                "--compress_full_confidence",
                compression_value,
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["compress_fold_input"] is False
        assert captured["compress_full_confidence"] is expected

    help_result = CliRunner().invoke(batch_inference.predict, ["--help"])
    assert help_result.exit_code == 0
    assert "--compress_fold_input" in help_result.output
    assert "--compress_full_confidence" in help_result.output


def test_prep_forwards_compress_fold_input(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        batch_inference,
        "prepare_input_jobs",
        lambda *_args, **kwargs: captured.update(kwargs) or [],
    )

    result = CliRunner().invoke(
        batch_inference.inputprep,
        ["--input", str(tmp_path / "input.json"), "--compress_fold_input", "false"],
    )

    assert result.exit_code == 0, result.output
    assert captured["compress_fold_input"] is False
    help_result = CliRunner().invoke(batch_inference.inputprep, ["--help"])
    assert "--compress_fold_input" in help_result.output


@pytest.mark.parametrize(
    "writer",
    [
        lambda path: path.write_bytes(b"not an npz"),
        lambda path: np.savez_compressed(path),
        lambda path: np.savez_compressed(path, bad=np.array([{"x": 1}], dtype=object)),
    ],
)
def test_nonempty_npz_content_is_not_validated(tmp_path, writer):
    _write_complete_npz_seed(tmp_path, "job", 11, 1)
    writer(_sample_paths(tmp_path, "job", 11, 0)["full_npz"])

    assert seed_outputs_complete(
        tmp_path,
        "job",
        11,
        1,
        need_atom_confidence=True,
        compress_full_confidence=True,
    )
