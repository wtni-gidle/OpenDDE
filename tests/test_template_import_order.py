"""Template modules must be usable without pre-importing the public runner."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("module", [
    "opendde.data.template.template_featurizer",
    "opendde.data.template.template_finalizer",
    "opendde.data.inference.infer_dataloader",
])
def test_template_import_in_fresh_interpreter(module):
    # A separate interpreter prevents other tests from masking an import cycle.
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
