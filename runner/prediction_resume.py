# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Lightweight completeness checks for AF3-style prediction outputs."""

from __future__ import annotations

import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from opendde.data.inference.input_validation import (
    validate_inference_seed,
    validate_sample_name,
)


def _contained_nonempty_file(path: Path, job_dir: Path) -> Path | None:
    """Resolve a non-empty regular file that remains inside its job directory."""
    try:
        resolved_path = path.resolve(strict=True)
        resolved_job_dir = job_dir.resolve(strict=True)
        if resolved_job_dir not in resolved_path.parents:
            return None
        file_stat = resolved_path.stat()
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size <= 0:
            return None
        return resolved_path
    except OSError:
        return None


def seed_outputs_complete(
    output_dir: str | Path,
    job_name: str,
    seed: int,
    num_samples: int,
    *,
    need_atom_confidence: bool,
    compress_full_confidence: bool = False,
) -> bool:
    """Check canonical outputs using only file metadata, never their contents.

    Existing outputs may be reused even when inputs or model settings change.
    """
    if isinstance(num_samples, bool) or not isinstance(num_samples, int):
        return False
    if num_samples <= 0:
        return False
    try:
        safe_name = validate_sample_name(job_name)
        safe_seed = validate_inference_seed(seed)
        output_root = Path(output_dir).expanduser().resolve()
        job_dir = (output_root / safe_name).resolve()
        if output_root not in job_dir.parents:
            return False

        for sample_index in range(num_samples):
            prefix = f"seed-{safe_seed}_sample-{sample_index}"
            if _contained_nonempty_file(
                job_dir / "models" / f"{prefix}_model.cif",
                job_dir,
            ) is None:
                return False
            if _contained_nonempty_file(
                job_dir / "summary_confidences" / f"{prefix}_summary_confidences.json",
                job_dir,
            ) is None:
                return False
            if need_atom_confidence:
                full_path = (
                    job_dir
                    / "full_data"
                    / (
                        f"{prefix}_full_data.npz"
                        if compress_full_confidence
                        else f"{prefix}_full_data.json"
                    )
                )
                if _contained_nonempty_file(full_path, job_dir) is None:
                    return False
    except (OSError, TypeError, ValueError):
        return False
    return True


def incomplete_job_seed_schedule(
    output_dir: str | Path,
    jobs: Sequence[dict[str, Any]],
    job_seed_schedule: Sequence[Sequence[int]],
    num_samples: int,
    *,
    need_atom_confidence: bool,
    compress_full_confidence: bool = False,
) -> list[list[int]]:
    """Return incomplete seeds for each job without changing requested order."""
    if len(jobs) != len(job_seed_schedule):
        raise ValueError("Each inference job must have one seed schedule entry.")
    return [
        [
            seed
            for seed in seeds
            if not seed_outputs_complete(
                output_dir,
                job["name"],
                seed,
                num_samples,
                need_atom_confidence=need_atom_confidence,
                compress_full_confidence=compress_full_confidence,
            )
        ]
        for job, seeds in zip(jobs, job_seed_schedule, strict=True)
    ]
