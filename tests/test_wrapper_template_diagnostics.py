"""Template preparation exposes failures, filtering, and retained resources."""

import json
import logging
from pathlib import Path

import pytest

from opendde.data.template.template_finalizer import finalize_template_hits
from opendde.data.template.template_parser import TemplateHit, TemplateSearchResult
from opendde.data.template.template_utils import TemplateHitFeaturizer
from runner.template_search import update_template_info
from tests.test_explicit_templates import _cif


LOGGER = "opendde.data.template.template_finalizer"
QUERY = "CAAAAAAAAAAA"
HIT = ">tiny_A/1-12 mol:protein length:12\nAAAAAAAAAAAA\n"


def _online(tmp_path, *, known_date=True):
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps({"tiny": {"release_date": "1999-01-01"}} if known_date else {})
    )
    return TemplateHitFeaturizer(
        mmcif_dir=str(tmp_path),
        release_dates_path=str(release),
        max_hits=4,
        fetch_remote=False,
    )


def _messages(caplog, level):
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == LOGGER and record.levelno == level
    ]


@pytest.mark.parametrize(
    "known_date,reason",
    [(True, "CIF load failed for tiny"), (False, "tiny date unknown")],
)
def test_failed_templates_report_reason_and_chain_but_continue(
    tmp_path, caplog, known_date, reason
):
    # Dropping native errors or raising instead of publishing [] breaks this contract.
    path = tmp_path / "hits.a3m"
    path.write_text(HIT)
    jobs = [{"name": "diagnostics-job", "sequences": [
        {"rnaSequence": {"sequence": "AU"}},
        {"proteinChain": {"sequence": QUERY, "id": ["B", "C"],
                          "templatesPath": str(path)}},
    ]}]

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert update_template_info(
            jobs, template_featurizer=_online(tmp_path, known_date=known_date),
            max_template_date="2000-01-01",
        )

    assert jobs[0]["sequences"][1]["proteinChain"] == {
        "sequence": QUERY, "id": ["B", "C"], "templates": [],
    }
    errors = _messages(caplog, logging.ERROR)
    assert len(errors) == 1
    assert reason in errors[0]
    for field in ("diagnostics-job", "sequences[1]", "B", "C", str(path)):
        assert field in errors[0]
    summary = "\n".join(_messages(caplog, logging.INFO))
    assert "retained 0 templates: []" in summary
    assert "errors=1" in summary
    assert "continuing without templates" in summary


@pytest.mark.parametrize(
    "content,outcome,warning",
    [("", "no search hits", None),
     (">tiny_A/1-7 mol:protein length:7\nAAAAAAA\n",
      "no templates retained after filtering/selection", "Template sequence too short")],
)
def test_normal_zero_templates_are_distinct_from_processing_errors(
    tmp_path, caplog, content, outcome, warning
):
    # Treating a normal empty result as an error or losing filter warnings is a bug.
    path = tmp_path / "hits.a3m"
    path.write_text(content)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        entries = finalize_template_hits(
            QUERY, path, _online(tmp_path), max_template_date="2000-01-01"
        )

    assert entries == []
    assert not _messages(caplog, logging.ERROR)
    summary = "\n".join(_messages(caplog, logging.INFO))
    assert "retained 0 templates: []" in summary
    assert "errors=0" in summary
    assert outcome in summary
    warnings = _messages(caplog, logging.WARNING)
    if warning is None:
        assert warnings == []
    else:
        assert len(warnings) == 1
        assert warning in warnings[0]
        assert str(path) in warnings[0]


def test_partial_success_reports_diagnostics_and_only_exported_identities(
    tmp_path, monkeypatch, caplog
):
    # Logging all selected hits would misreport the four-template export limit.
    chains = ("A", "B", "C", "D", "E")
    (tmp_path / "tiny.cif").write_text(_cif(chains))
    path = tmp_path / "hits.a3m"
    path.write_text("".join(
        f">tiny_{chain}/1-7 mol:protein length:7\nAAAAAAA\n" for chain in chains
    ))
    online = _online(tmp_path)
    hits = [TemplateHit(index, f"tiny_{chain}", 7, None, "AAAAAAA", "AAAAAAA",
                        list(range(7)), list(range(7)))
            for index, chain in enumerate(chains, 1)]
    # Substitute the alignment/feature stage; parsing and CIF export stay real.
    result = TemplateSearchResult(
        [{"template_sequence": b"AAAAAAA"} for _ in hits], hits,
        ["Error processing hit: kalign unavailable", "other1 date unknown."],
        ["Hit skip_A failed prefilter: Template sequence too short.",
         "Hit skip_B failed prefilter: Hit is a large duplicate of the query."],
    )
    monkeypatch.setattr(online, "get_templates", lambda **kwargs: (result, {}))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        entries = finalize_template_hits(
            QUERY, path, online, max_template_date="2000-01-01"
        )

    assert [Path(entry["mmcifPath"]).name for entry in entries] == [
        "tiny_A.cif", "tiny_B.cif", "tiny_C.cif", "tiny_D.cif",
    ]
    assert all(Path(entry["mmcifPath"]).is_file() for entry in entries)
    errors = _messages(caplog, logging.ERROR)
    warnings = _messages(caplog, logging.WARNING)
    assert len(errors) == len(warnings) == 2
    assert all(any(reason in message for message in errors) for reason in result.errors)
    assert all(any(reason in message for message in warnings) for reason in result.warnings)
    summary = "\n".join(_messages(caplog, logging.INFO))
    assert "retained 4 templates:" in summary
    assert all(name in summary for name in ("tiny_A", "tiny_B", "tiny_C", "tiny_D"))
    assert "tiny_E" not in summary
    assert "errors=2" in summary and "warnings=2" in summary
