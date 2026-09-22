"""Prepared MSA replacement acceptance at native inference consumer boundaries.

The tiny dataloader adapter supplies annotated atoms instead of building CCD atom
features or running a model. JSON loading/resolution, MSA parsing, species pairing,
deduplication, explicit mmCIF extraction, and template assembly remain real.
"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray
from ml_collections import ConfigDict

# Enter through the public runner, as the CLI does, before template imports.
from runner import batch_inference, inference, msa_search, template_search
from opendde.data.msa.msa_featurizer import InferenceMSAFeaturizer
from opendde.data.template.template_featurizer import InferenceTemplateFeaturizer
from opendde.utils.text_io import read_text, write_zstd_text_atomic
from tests.test_explicit_templates import _cif


OLD_UNPAIRED = ">query\nAAAAAAA\n>old_condition\nDAAAAAA\n"
NEW_UNPAIRED = ">query\nAAAAAAA\n>replacement_condition\nWaaAAAAAA\n"
PAIRED = (
    ">query\nAAAAAAA\n>sp|P12345|PAIR_HUMAN\nRAAAAAA\n",
    ">query\nCCCCCCC\n>sp|P67890|PAIR_HUMAN\nRCCCCCC\n",
)
MAPPING = {"queryIndices": [0, 1, 2, 4, 6], "templateIndices": [1, 2, 3, 4, 5]}


def _snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _atoms():
    atoms = AtomArray(14)
    atoms.chain_id[:] = ["A"] * 7 + ["B"] * 7
    atoms.res_id[:] = list(range(1, 8)) * 2
    atoms.set_annotation("asym_id_int", np.repeat([0, 1], 7))
    atoms.set_annotation("centre_atom_mask", np.ones(14, dtype=bool))
    return atoms


def _prepare_native_search_bundle(tmp_path, monkeypatch, compressed):
    """Replace only the external search service; native search orchestration runs."""
    (tmp_path / "manual.cif").write_text(_cif())
    source = tmp_path / "input.json"
    source.write_text(json.dumps([{
        "name": "replacement",
        "modelSeeds": [17],
        "sequences": [
            {"proteinChain": {
                "sequence": "AAAAAAA", "count": 1,
                "templates": [{"mmcifPath": "manual.cif", **MAPPING}],
            }},
            {"proteinChain": {
                "sequence": "CCCCCCC", "count": 1, "templates": [],
            }},
        ],
    }]))

    def search(seqs, out_dir, mode=None):
        assert seqs == ["AAAAAAA", "CCCCCCC"]
        result = []
        for index, unpaired in enumerate((
            OLD_UNPAIRED, ">query\nCCCCCCC\n>other_condition\nECCCCCC\n"
        )):
            root = Path(out_dir) / str(index)
            root.mkdir(parents=True)
            (root / "pairing.a3m").write_text(PAIRED[index])
            (root / "non_pairing.a3m").write_text(unpaired)
            result.append(str(root))
        return result

    monkeypatch.setattr(msa_search, "msa_search", search)
    monkeypatch.setattr(template_search, "update_template_info", _forbid_search)
    paths = batch_inference.run_prediction_workflow(
        str(source), str(tmp_path / "prepared"),
        run_data_pipeline=True, run_inference=False, write_input_json=True,
        use_template=True, compress_fold_input=compressed, skip=False,
    )
    return Path(paths[0])


def _forbid_search(*args, **kwargs):
    pytest.fail("Prepared conditions must not be replaced by automatic search")


def _install_native_consumers(monkeypatch, tmp_path, pair_as_unpair):
    observations = []

    class ConsumedLoader(list):
        dataset = ()

    def consumer_loader(configs, inputs):
        # infer_predict has loaded the public/private JSON and resolved paths.
        assert len(inputs) == 1
        chains = inputs[0]["sequences"]
        observations.append({
            "path": Path(configs.input_json_path),
            "chains": deepcopy(chains),
            "msa": InferenceMSAFeaturizer.make_msa_feature(
                chains, _atoms(), msa_pair_as_unpair=pair_as_unpair,
                use_rna_msa=False,
            ),
            "template": InferenceTemplateFeaturizer.make_template_feature(
                chains, _atoms(), use_template=True,
            ),
        })
        # No model batch: native inference completes its CPU control flow only.
        return ConsumedLoader()

    def runner_factory(**options):
        assert options["skip"] is False
        return SimpleNamespace(
            configs=ConfigDict({"seeds": [17], "deterministic": False, "skip": False}),
            foldcp_config=SimpleNamespace(enabled=False),
            device=torch.device("cpu"),
            error_dir=str(tmp_path / "errors"),
            close=lambda: None,
        )

    monkeypatch.setattr(batch_inference, "get_default_runner", runner_factory)
    monkeypatch.setattr(inference, "get_inference_dataloader", consumer_loader)
    monkeypatch.setattr(msa_search, "msa_search", _forbid_search)
    monkeypatch.setattr(batch_inference, "msa_search", _forbid_search)
    return observations


def _assert_native_condition(observation, replacement):
    features = observation["msa"]
    rows = features["msa"]
    # Native residue alphabet: A=0, R=1, D=3, C=4, W=17. Species pairing
    # must produce a joined, non-query row for the two distinct proteins.
    paired_row = [1, 0, 0, 0, 0, 0, 0, 1, 4, 4, 4, 4, 4, 4]
    assert np.any(np.all(rows == paired_row, axis=1))
    assert int(features["prot_paired_num_alignments"]) >= 2
    residue = 17 if replacement else 3
    matching = np.all(rows[:, :7] == [residue, 0, 0, 0, 0, 0, 0], axis=1)
    assert matching.any(), rows
    assert not np.any(rows[:, 0] == (3 if replacement else 17))
    assert np.all(features["deletion_matrix"][matching, 1] == (2 if replacement else 0))

    template = observation["template"]
    mask = template["template_atom_mask"]
    assert int(mask.sum()) == 5
    np.testing.assert_array_equal(np.where(mask[0].any(axis=-1))[0], [0, 1, 2, 4, 6])
    # Native extraction centers the seven CA coordinates before mapping them.
    np.testing.assert_allclose(
        template["template_atom_positions"][0][mask[0]][:, 0],
        [-7.6, -3.8, 0.0, 3.8, 7.6],
        atol=1e-5,
    )


@pytest.mark.parametrize("run_data_pipeline", [False, True])
@pytest.mark.parametrize("publish", [False, True])
@pytest.mark.parametrize("compressed,pair_as_unpair", [(True, True), (False, False)])
def test_same_path_unpaired_replacement_reaches_native_consumers_without_search(
    tmp_path, monkeypatch, run_data_pipeline, publish, compressed, pair_as_unpair,
):
    """Catch stale same-path reads, paired/template overwrite, or D/J coupling."""
    prepared = _prepare_native_search_bundle(tmp_path, monkeypatch, compressed)
    original_json = prepared.read_bytes()
    payload = json.loads(original_json)[0]
    chains = [entity["proteinChain"] for entity in payload["sequences"]]
    template_entry = chains[0]["templates"][0]
    assert {key: template_entry[key] for key in MAPPING} == MAPPING
    resource_paths = [
        chain[field] for chain in chains for field in ("pairedMsaPath", "unpairedMsaPath")
    ] + [template_entry["mmcifPath"]]
    assert all(not Path(path).is_absolute() for path in resource_paths)
    for index, chain in enumerate(chains):
        assert read_text(prepared.parent / chain["pairedMsaPath"]) == PAIRED[index]
    assert read_text(prepared.parent / template_entry["mmcifPath"]) == _cif()

    seen = _install_native_consumers(monkeypatch, tmp_path, pair_as_unpair)
    output = tmp_path / "prediction_output"
    output.mkdir()
    (output / "existing.txt").write_text("preserve public outputs")
    before_output = _snapshot(output)
    common = dict(use_template=True, use_rna_msa=False, skip=False)
    batch_inference.run_prediction_workflow(
        str(prepared), str(output), run_data_pipeline=False,
        write_input_json=False, **common,
    )
    assert len(seen) == 1
    _assert_native_condition(seen[0], replacement=False)

    unpaired = prepared.parent / chains[0]["unpairedMsaPath"]
    before_source = _snapshot(prepared.parent)
    if compressed:
        write_zstd_text_atomic(unpaired, NEW_UNPAIRED)
    else:
        unpaired.write_text(NEW_UNPAIRED)
    after_source = _snapshot(prepared.parent)
    assert {
        path for path in before_source if before_source[path] != after_source[path]
    } == {chains[0]["unpairedMsaPath"]}

    # Repeat at the identical path in one interpreter to expose retained caches.
    for _ in range(2):
        paths = batch_inference.run_prediction_workflow(
            str(prepared), str(output), run_data_pipeline=run_data_pipeline,
            write_input_json=publish, compress_fold_input=compressed, **common,
        )
        _assert_native_condition(seen[-1], replacement=True)
    assert len(seen) == 3
    assert prepared.read_bytes() == original_json
    assert _snapshot(prepared.parent) == after_source

    if not publish:
        assert paths == [str(prepared)]
        assert _snapshot(output) == before_output
    else:
        published = Path(paths[0])
        assert published == output / "replacement" / "replacement_data.json"
        rewritten = json.loads(published.read_text())[0]["sequences"]
        for index, entity in enumerate(rewritten):
            chain = entity["proteinChain"]
            for field in ("pairedMsaPath", "unpairedMsaPath"):
                assert not Path(chain[field]).is_absolute()
            assert read_text(published.parent / chain["pairedMsaPath"]) == PAIRED[index]
        chain = rewritten[0]["proteinChain"]
        assert read_text(published.parent / chain["unpairedMsaPath"]) == NEW_UNPAIRED
        assert chain["templates"] == chains[0]["templates"]
        assert read_text(published.parent / chain["templates"][0]["mmcifPath"]) == _cif()

    for observation in seen:
        chain = observation["chains"][0]["proteinChain"]
        assert Path(chain["unpairedMsaPath"]).is_absolute()
        assert {key: chain["templates"][0][key] for key in MAPPING} == MAPPING
