# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Tests for lightweight prediction resume behavior."""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner
import pytest
import torch

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
        for path in paths.values():
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


def _write_input(path: Path, jobs: list[dict]) -> Path:
    path.write_text(json.dumps(jobs), encoding="utf-8")
    return path


def test_seed_outputs_require_each_canonical_sample_and_optional_full_data(tmp_path):
    _write_complete_seed(tmp_path, "job", 7, 2, full_data=False)

    assert seed_outputs_complete(tmp_path, "job", 7, 2, need_atom_confidence=False)
    assert not seed_outputs_complete(tmp_path, "job", 7, 3, need_atom_confidence=False)
    assert not seed_outputs_complete(tmp_path, "job", 7, 2, need_atom_confidence=True)


@pytest.mark.parametrize(
    ("target", "contents"),
    [
        ("model", b""),
        ("summary", b""),
        ("summary", b"not-json"),
        ("summary", b"{}"),
        ("full", b""),
        ("full", b"[]"),
        ("full", b"{}"),
    ],
)
def test_empty_or_corrupt_required_output_is_incomplete(tmp_path, target, contents):
    _write_complete_seed(tmp_path, "job", 8, 1)
    _sample_paths(tmp_path, "job", 8, 0)[target].write_bytes(contents)

    assert not seed_outputs_complete(tmp_path, "job", 8, 1, need_atom_confidence=True)


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


def test_write_now_false_warns_once_and_still_allows_pre_runner_skip(
    tmp_path, monkeypatch, caplog
):
    source = _write_input(
        tmp_path / "job_data.json",
        [{"name": "job", "modelSeeds": [7], "sequences": []}],
    )
    output = tmp_path / "output"
    _write_complete_seed(output, "job", 7, 1, full_data=False)
    monkeypatch.setattr(
        batch_inference,
        "get_default_runner",
        lambda **_kwargs: pytest.fail("complete outputs must not initialize runner"),
    )

    with caplog.at_level(logging.WARNING):
        batch_inference.run_prediction_workflow(
            str(source),
            str(output),
            run_data_pipeline=False,
            run_inference=True,
            n_sample=1,
            need_atom_confidence=False,
            skip=True,
            write_now=False,
        )

    messages = [record.getMessage() for record in caplog.records]
    assert (
        sum("always writes each prediction synchronously" in msg for msg in messages)
        == 1
    )


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
        write_now=True,
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
        write_now=True,
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


def test_direct_write_now_warning_is_emitted_once(tmp_path, monkeypatch, caplog):
    source = _write_input(
        tmp_path / "job_data.json",
        [{"name": "job", "modelSeeds": [7], "sequences": []}],
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
        write_now=False,
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

    with caplog.at_level(logging.WARNING):
        inference.infer_predict(runner, configs)
        inference.infer_predict(runner, configs)

    messages = [record.getMessage() for record in caplog.records]
    assert (
        sum("always writes each prediction synchronously" in msg for msg in messages)
        == 1
    )


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        ([], (False, True)),
        (["--skip", "true", "--write_now", "false"], (True, False)),
    ],
)
def test_cli_forwards_skip_and_write_now(tmp_path, monkeypatch, extra_args, expected):
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
    assert (captured["skip"], captured["write_now"]) == expected
