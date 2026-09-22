# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Validation shared by inference loading and output path construction."""

from __future__ import annotations

from typing import Any
from pathlib import Path

_MAX_NUMPY_SEED = 2**32 - 1


def validate_resource_choice(data, field, *, required=False):
    inline, path = data.get(field), data.get(field + "Path")
    if inline is not None and path is not None:
        raise ValueError(f"Use only {field} or {field}Path, not both.")
    if inline is not None and not isinstance(inline, str):
        raise ValueError(f"{field} must be text.")
    if path is not None and (not isinstance(path, str) or not path):
        raise ValueError(f"{field}Path must be a non-empty path.")
    if required and inline is None and path is None:
        raise ValueError(f"Provide {field} or {field}Path.")


def has_explicit_msa(chain, channels=("pairedMsa", "unpairedMsa")):
    """Empty inline text is explicit; invalid supplied paths never trigger search."""
    supplied = False
    for channel in channels:
        validate_resource_choice(chain, channel)
        supplied |= (
            chain.get(channel) is not None or chain.get(channel + "Path") is not None
        )
        path = chain.get(channel + "Path")
        if path is not None and not Path(path).is_file():
            raise FileNotFoundError(f"{channel}Path does not exist: {path}")
    return supplied


def validate_template_entry(entry, query_length, template_length=None):
    if not isinstance(entry, dict):
        raise ValueError("Each templates entry must be an object.")
    if "chainId" in entry:
        raise ValueError(
            "An explicit template does not support chainId; use single-chain mmCIF."
        )
    validate_resource_choice(entry, "mmcif", required=True)
    for key, limit in (
        ("queryIndices", query_length),
        ("templateIndices", template_length),
    ):
        values = entry.get(key)
        if not isinstance(values, list) or any(
            type(i) is not int or i < 0 for i in values
        ):
            raise ValueError(f"{key} indices must be nonnegative integers.")
        if len(set(values)) != len(values):
            raise ValueError(f"{key} indices must be unique.")
        if limit is not None and any(i >= limit for i in values):
            raise ValueError(f"{key} index exceeds sequence length {limit}.")
    if len(entry["queryIndices"]) != len(entry["templateIndices"]):
        raise ValueError("queryIndices and templateIndices must have equal lengths.")


def validate_chain_conditions(chain, *, protein=True):
    if "templatesPath" in chain:
        raise ValueError(
            "templatesPath is no longer supported; use templates in the main JSON."
        )
    for channel in ("pairedMsa", "unpairedMsa") if protein else ("unpairedMsa",):
        validate_resource_choice(chain, channel)
    if protein and chain.get("templates") is not None:
        if not isinstance(chain["templates"], list):
            raise ValueError("templates must be a list.")
        for entry in chain["templates"]:
            validate_template_entry(entry, len(chain.get("sequence", "")))


def validate_inference_seed(value: Any, *, location: str = "seed") -> int:
    """Validate a seed before it reaches NumPy/PyTorch RNG setup."""

    if isinstance(value, bool):
        raise ValueError(f"{location} must be an integer, not a boolean.")
    if isinstance(value, int):
        seed = value
    elif isinstance(value, str) and value.strip() == value and value.isdecimal():
        # Preserve compatibility with older JSON/CLI inputs that quoted seeds.
        seed = int(value)
    else:
        raise ValueError(f"{location} must be an integer; got {value!r}.")
    if not 0 <= seed <= _MAX_NUMPY_SEED:
        raise ValueError(f"{location} must be in [0, {_MAX_NUMPY_SEED}]; got {seed}.")
    return seed


def validate_sample_name(value: Any, *, job_index: int | None = None) -> str:
    """Return a safe output path component for one inference job."""

    location = f" for job {job_index}" if job_index is not None else ""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Inference job name{location} must be a non-empty string.")
    if value.casefold() == "err":
        raise ValueError(
            f"Inference job name{location} {value!r} is reserved for error reports."
        )
    if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(
            f"Inference job name{location} must be a single safe path component; "
            f"got {value!r}."
        )
    return value


def validate_inference_jobs(value: Any) -> list[dict[str, Any]]:
    """Validate the top-level inference job list and collision-sensitive fields."""

    if not isinstance(value, list) or not value:
        raise ValueError(
            "Input JSON must be a non-empty top-level list, "
            f"got {type(value).__name__}."
        )

    jobs: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, job in enumerate(value):
        if not isinstance(job, dict):
            raise ValueError(
                f"Inference job {index} must be an object, got {type(job).__name__}."
            )
        name = validate_sample_name(job.get("name"), job_index=index)
        if name in seen_names:
            raise ValueError(
                f"Inference job name {name!r} is duplicated in the same input JSON; "
                "duplicate names would overwrite outputs."
            )
        seen_names.add(name)
        for sequence in job.get("sequences", []):
            for kind in ("proteinChain", "rnaSequence"):
                if kind in sequence:
                    validate_chain_conditions(
                        sequence[kind], protein=kind == "proteinChain"
                    )

        model_seeds = job.get("modelSeeds")
        if model_seeds is not None:
            if not isinstance(model_seeds, list):
                raise ValueError(
                    f"modelSeeds for job {name!r} must be a list of integers."
                )
            for seed in model_seeds:
                try:
                    validate_inference_seed(
                        seed,
                        location=f"modelSeeds for job {name!r}",
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"modelSeeds for job {name!r} contains invalid seed {seed!r}."
                    ) from exc
        jobs.append(job)
    return jobs
