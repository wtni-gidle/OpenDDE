"""Prepared wrappers read current CIFs without parsed-template caches."""
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from opendde.data.template.template_finalizer import finalize_template_hits
from opendde.data.template.template_parser import TemplateHit
from opendde.data.template.template_utils import TemplateHitFeaturizer
from runner import template_search
from runner.fold_input import prepare_input_jobs
from tests.test_explicit_templates import _cif


def input_file(tmp_path, *, templates=None):
    source = tmp_path / "input.json"
    source.write_text(json.dumps([{"name": "job", "sequences": [{"proteinChain": {
        "sequence": "CAAAAAA", "count": 1, "pairedMsa": "",
        "unpairedMsa": ">q\nCAAAAAA\n", "templates": templates}}]}]))
    return source


def test_default_data_reads_replaced_cif_without_creating_parse_cache(tmp_path, monkeypatch):
    # Other configuration tests reload this module; use its current dictionary.
    from opendde.config.data import data_configs

    kalign = shutil.which("kalign")
    if kalign is None:
        pytest.skip("Kalign is required for this real template processing test")
    source = input_file(tmp_path)
    cif = tmp_path / "tiny.cif"
    cif.write_text(_cif())
    release = tmp_path / "release.json"
    release.write_text("{}")
    for key, value in dict(prot_template_mmcif_dir=str(tmp_path),
        release_dates_path=str(release), obsolete_pdbs_path=None,
        kalign_binary_path=kalign, fetch_remote=False).items():
        monkeypatch.setitem(data_configs["template"], key, value)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("SLURM_TMPDIR", str(scratch))
    observations = []

    def consume_instead_of_database_search(jobs, *, template_featurizer, **kwargs):
        # Replace only database search; consume the real factory-created processor,
        # actual CIF parser, Kalign, and coordinate extractor while scratch is alive.
        hit = TemplateHit(1, "tiny_A", 7, None, "CAAAAAA", "AAAAAAA",
                          list(range(7)), list(range(7)))
        processor = template_featurizer._hit_processor
        assert processor._fetch_remote is False, "This test must remain offline"
        old, _ = processor.process("CAAAAAA", hit, datetime(2100, 1, 1), {}, {})
        assert old.features is not None, old.error
        cif.write_text(_cif().replace("3.8 0 0", "4.8 0 0"))
        new, _ = processor.process("CAAAAAA", hit, datetime(2100, 1, 1), {}, {})
        assert new.features is not None, new.error
        assert not np.array_equal(old.features["template_all_atom_positions"],
                                  new.features["template_all_atom_positions"])
        assert not list(scratch.rglob("*.pkl"))
        observations.append(True)
        jobs[0]["sequences"][0]["proteinChain"]["templates"] = []
        return True

    monkeypatch.setattr(template_search, "update_template_info", consume_instead_of_database_search)
    [prepared] = prepare_input_jobs(str(source), str(tmp_path / "out"),
        use_msa=False, use_template=True, max_template_date="2100-01-01")
    assert observations == [True]
    assert Path(prepared).is_file()
    assert not list(scratch.iterdir())


def test_preparation_rejects_cached_custom_processor_before_publication(tmp_path):
    source = input_file(tmp_path, templates=[])
    online = TemplateHitFeaturizer(mmcif_dir=str(tmp_path),
                                  template_cache_dir=str(tmp_path / "cache"))
    with pytest.raises(ValueError, match="template_cache_dir"):
        prepare_input_jobs(str(source), str(tmp_path / "out"), use_msa=False,
                           use_template=True, template_featurizer=online)
    assert not (tmp_path / "out").exists()


def test_finalizer_rejects_cached_custom_processor(tmp_path):
    online = TemplateHitFeaturizer(mmcif_dir=str(tmp_path),
                                  template_cache_dir=str(tmp_path / "cache"))
    hits = tmp_path / "hits.a3m"
    hits.write_text(">tiny_A/1-7 mol:protein length:7\nAAAAAAA\n")
    with pytest.raises(ValueError, match="template_cache_dir"):
        finalize_template_hits("CAAAAAA", hits, online, max_template_date="2100-01-01")
