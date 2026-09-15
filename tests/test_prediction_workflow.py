# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from runner import batch_inference, fold_input, inference, msa_search, template_search


def write_input(tmp_path, jobs=None):
    path = tmp_path / "input.json"
    path.write_text(
        json.dumps(
            jobs
            or [
                {
                    "name": "job",
                    "sequences": [{"proteinChain": {"sequence": "ACD", "count": 1}}],
                }
            ]
        )
    )
    return path


def forbid(*args, **kwargs):
    pytest.fail("unexpected model initialization or data-stage call")


def test_data_only_searches_and_writes_one_portable_json(tmp_path, monkeypatch):
    source = write_input(tmp_path)
    output = tmp_path / "output"
    scratches = []

    def search(seqs, msa_res_dir, mode=None):
        assert seqs == ["ACD"]
        scratch = Path(msa_res_dir)
        scratches.append(scratch)
        scratch.mkdir(parents=True)
        (scratch / "non_pairing.a3m").write_text(">query\nACD\n")
        return [str(scratch)]

    monkeypatch.setattr(msa_search, "msa_search", search)
    monkeypatch.setattr(batch_inference, "get_default_runner", forbid)
    monkeypatch.setattr(batch_inference, "preprocess_input", forbid)
    paths = batch_inference.run_prediction_workflow(
        str(source), str(output), run_data_pipeline=True, run_inference=False
    )
    assert paths == [str(output / "job" / "job_data.json")]
    assert list(output.rglob("*.json")) == [Path(paths[0])]
    assert not (output / ".opendde_preprocessed").exists()
    chain = json.loads(Path(paths[0]).read_text())[0]["sequences"][0]["proteinChain"]
    assert chain["unpairedMsaPath"] == "msas/job__A_unpairedmsa.a3m"
    assert (
        Path(paths[0]).parent / chain["unpairedMsaPath"]
    ).read_text() == ">query\nACD\n"
    assert scratches and all(not path.exists() for path in scratches)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"run_data_pipeline": False, "run_inference": False}, "at least one"),
        (
            {
                "run_data_pipeline": True,
                "run_inference": True,
                "foldcp_mode": "distributed",
            },
            "-D false -P true",
        ),
    ],
)
def test_invalid_stage_combinations_fail_without_outputs(
    tmp_path, monkeypatch, options, message
):
    source = write_input(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(batch_inference, "get_default_runner", forbid)
    with pytest.raises(ValueError, match=message):
        batch_inference.run_prediction_workflow(str(source), str(output), **options)
    assert not output.exists()


def test_inference_only_preserves_exact_prepared_path_and_closes_on_failure(
    tmp_path, monkeypatch
):
    source = write_input(tmp_path)
    closed = []
    runner = SimpleNamespace(configs={}, close=lambda: closed.append(True))
    monkeypatch.setattr(batch_inference, "get_default_runner", lambda **kwargs: runner)
    for name in (
        "preprocess_input",
        "update_infer_json",
        "update_template_info",
        "prepare_input_jobs",
    ):
        monkeypatch.setattr(batch_inference, name, forbid, raising=False)

    def predict(actual_runner, configs):
        assert actual_runner is runner
        assert configs["input_json_path"] == str(source)
        raise RuntimeError("prediction failed")

    monkeypatch.setattr(batch_inference, "infer_predict", predict)
    with pytest.raises(RuntimeError, match="prediction failed"):
        batch_inference.run_prediction_workflow(
            str(source),
            str(tmp_path / "out"),
            run_data_pipeline=False,
            run_inference=True,
            foldcp_mode="distributed",
        )
    assert closed == [True]


def test_all_data_prepared_before_runner_initialization(tmp_path, monkeypatch):
    source = write_input(
        tmp_path,
        [{"name": "first", "sequences": []}, {"name": "second", "sequences": []}],
    )
    output = tmp_path / "output"
    predictions = []

    def make_runner(**kwargs):
        assert (output / "first" / "first_data.json").exists()
        assert (output / "second" / "second_data.json").exists()
        return SimpleNamespace(configs={}, close=lambda: None)

    monkeypatch.setattr(batch_inference, "get_default_runner", make_runner)
    monkeypatch.setattr(
        batch_inference,
        "infer_predict",
        lambda runner, configs: predictions.append(configs["input_json_path"]),
    )
    paths = batch_inference.run_prediction_workflow(
        str(source), str(output), use_msa=False
    )
    assert predictions == paths
    assert len(paths) == 2


def test_inference_directory_can_share_output_and_ignores_prediction_json(
    tmp_path, monkeypatch
):
    source = write_input(tmp_path)
    prepared = tmp_path / "job_data.json"
    source.rename(prepared)
    (tmp_path / "confidence.json").write_text('{"score": 0.8}')
    predictions = []
    monkeypatch.setattr(
        batch_inference,
        "get_default_runner",
        lambda **kwargs: SimpleNamespace(configs={}, close=lambda: None),
    )
    monkeypatch.setattr(
        batch_inference,
        "infer_predict",
        lambda runner, configs: predictions.append(configs["input_json_path"]),
    )
    paths = batch_inference.run_prediction_workflow(
        str(tmp_path), str(tmp_path), run_data_pipeline=False
    )
    assert predictions == paths == [str(prepared)]


@pytest.mark.parametrize(
    "templates",
    [[], [{"mmcifPath": "manual.cif", "queryIndices": [0], "templateIndices": [0]}]],
)
def test_explicit_templates_skip_automatic_path(tmp_path, monkeypatch, templates):
    (tmp_path / "manual.cif").write_text("data_manual\n")
    source = write_input(
        tmp_path,
        [
            {
                "name": "job",
                "sequences": [
                    {
                        "proteinChain": {
                            "sequence": "ACD",
                            "count": 1,
                            "templates": templates,
                        }
                    }
                ],
            }
        ],
    )
    monkeypatch.setattr(template_search, "update_template_info", forbid)
    paths = fold_input.prepare_input_jobs(
        str(source), str(tmp_path / "out"), use_msa=False, use_template=True
    )
    prepared_templates = json.loads(Path(paths[0]).read_text())[0]["sequences"][0][
        "proteinChain"
    ]["templates"]
    if templates:
        assert prepared_templates == [
            {
                "mmcifPath": "msas/job__A_template_0.cif",
                "queryIndices": [0],
                "templateIndices": [0],
            }
        ]
        assert (
            Path(paths[0]).parent / prepared_templates[0]["mmcifPath"]
        ).read_text() == "data_manual\n"
    else:
        assert prepared_templates == []


def test_template_cutoff_reaches_finalizer_and_search_uses_scratch(
    tmp_path, monkeypatch
):
    source = write_input(
        tmp_path,
        [
            {
                "name": "job",
                "sequences": [
                    {
                        "proteinChain": {
                            "sequence": "ACD",
                            "count": 1,
                            "unpairedMsaPath": "custom.a3m",
                        }
                    }
                ],
            }
        ],
    )
    (tmp_path / "custom.a3m").write_text(">query\nACD\n")
    seen = []

    def search(**kwargs):
        assert Path(kwargs["msa_for_template_search_dir"]) != tmp_path
        Path(kwargs["output_path"]).write_text("searched hits")

    def finalize(sequence, path, featurizer, *, max_template_date):
        assert Path(path).read_text() == "searched hits"
        seen.append(max_template_date)
        return []

    monkeypatch.setattr(template_search, "run_template_search", search)
    monkeypatch.setattr(template_search, "finalize_template_hits", finalize)
    paths = fold_input.prepare_input_jobs(
        str(source),
        str(tmp_path / "out"),
        use_msa=False,
        use_template=True,
        max_template_date="2040-01-02",
        template_featurizer=object(),
    )
    assert seen == ["2040-01-02"]
    chain = json.loads(Path(paths[0]).read_text())[0]["sequences"][0]["proteinChain"]
    assert chain["templates"] == []
    assert "templatesPath" not in chain
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "custom.a3m",
        "input.json",
        "out",
    ]


def test_inference_resolves_prepared_resources_before_dataset(tmp_path, monkeypatch):
    source = write_input(
        tmp_path,
        [
            {
                "name": "job",
                "modelSeeds": [1],
                "sequences": [
                    {
                        "proteinChain": {
                            "sequence": "ACD",
                            "count": 1,
                            "unpairedMsaPath": "msas/a.a3m",
                            "templates": [
                                {
                                    "mmcifPath": "msas/a.cif",
                                    "queryIndices": [0],
                                    "templateIndices": [0],
                                }
                            ],
                        }
                    }
                ],
            }
        ],
    )
    seen = []

    def dataset(configs, jobs, *args):
        seen.extend(jobs)
        raise RuntimeError("stop before model execution")

    monkeypatch.setattr(inference, "_create_inference_dataloader_synchronized", dataset)
    runner = SimpleNamespace(
        foldcp_config=SimpleNamespace(enabled=False), error_dir=str(tmp_path / "errors")
    )
    configs = SimpleNamespace(input_json_path=str(source), seeds=[1])
    with pytest.raises(RuntimeError, match="stop before model execution"):
        inference.infer_predict(runner, configs)
    chain = seen[0]["sequences"][0]["proteinChain"]
    assert chain["unpairedMsaPath"] == str(tmp_path / "msas" / "a.a3m")
    assert chain["templates"][0]["mmcifPath"] == str(tmp_path / "msas" / "a.cif")
    assert list(tmp_path.rglob("*.json")) == [source]
