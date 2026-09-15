# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Prepare JSON inference jobs with portable local resource bundles."""

from __future__ import annotations

from copy import deepcopy
import json
from os import PathLike
from pathlib import Path
import shutil
from typing import Any

from opendde.data.inference.input_validation import validate_inference_jobs


def _resolve_path(path: str, json_path: Path) -> str:
    resource_path = Path(path)
    if resource_path.is_absolute():
        return str(resource_path)
    return str(json_path.parent / resource_path)


def _resolve_chain_paths(chain: dict[str, Any], json_path: Path) -> None:
    for field in ("pairedMsaPath", "unpairedMsaPath", "templatesPath"):
        if isinstance(chain.get(field), str):
            chain[field] = _resolve_path(chain[field], json_path)
    templates = chain.get("templates")
    if isinstance(templates, list):
        for template in templates:
            if isinstance(template, dict) and isinstance(
                template.get("mmcifPath"), str
            ):
                template["mmcifPath"] = _resolve_path(template["mmcifPath"], json_path)


def resolve_job_paths(job: dict[str, Any], json_path: Path) -> dict[str, Any]:
    """Return a copy of *job* whose supported resource paths are absolute."""
    resolved = deepcopy(job)
    sequences = resolved.get("sequences")
    if not isinstance(sequences, list):
        return resolved
    for sequence in sequences:
        if not isinstance(sequence, dict):
            continue
        protein = sequence.get("proteinChain")
        if isinstance(protein, dict):
            _resolve_chain_paths(protein, json_path)
        rna = sequence.get("rnaSequence")
        if isinstance(rna, dict) and isinstance(rna.get("unpairedMsaPath"), str):
            rna["unpairedMsaPath"] = _resolve_path(rna["unpairedMsaPath"], json_path)
    return resolved


def load_input_jobs(input_path: str) -> list[tuple[Path, dict[str, Any]]]:
    """Load and validate jobs, resolving supported relative paths from the JSON file."""
    json_path = Path(input_path).resolve()
    with json_path.open(encoding="utf-8") as handle:
        jobs = validate_inference_jobs(json.load(handle))
    return [(json_path, resolve_job_paths(job, json_path)) for job in jobs]


def _entity_label(chain: dict[str, Any], index: int) -> str:
    entity_ids = chain.get("id")
    if isinstance(entity_ids, list) and entity_ids:
        return str(entity_ids[0])
    return chr(ord("A") + index)


def _copy_resource(
    chain: dict[str, Any], field: str, destination: Path, job_dir: Path
) -> None:
    source = chain.get(field)
    if not isinstance(source, str):
        return
    shutil.copyfile(source, destination)
    chain[field] = destination.relative_to(job_dir).as_posix()


def write_prepared_job(job: dict[str, Any], out_dir: str | PathLike[str]) -> str:
    """Copy bundle resources for one job and write its single-item input JSON."""
    prepared = deepcopy(validate_inference_jobs([job])[0])
    job_dir = Path(out_dir) / prepared["name"]
    msa_dir = job_dir / "msas"
    msa_dir.mkdir(parents=True, exist_ok=True)

    sequences = prepared.get("sequences")
    if isinstance(sequences, list):
        for index, sequence in enumerate(sequences):
            if not isinstance(sequence, dict):
                continue
            protein = sequence.get("proteinChain")
            if isinstance(protein, dict):
                label = _entity_label(protein, index)
                _copy_resource(
                    protein,
                    "pairedMsaPath",
                    msa_dir / f"{prepared['name']}__{label}_pairedmsa.a3m",
                    job_dir,
                )
                _copy_resource(
                    protein,
                    "unpairedMsaPath",
                    msa_dir / f"{prepared['name']}__{label}_unpairedmsa.a3m",
                    job_dir,
                )
                templates = protein.get("templates")
                if isinstance(templates, list):
                    for template_index, template in enumerate(templates):
                        if not isinstance(template, dict):
                            continue
                        _copy_resource(
                            template,
                            "mmcifPath",
                            msa_dir
                            / f"{prepared['name']}__{label}_template_{template_index}.cif",
                            job_dir,
                        )
            rna = sequence.get("rnaSequence")
            if isinstance(rna, dict):
                label = _entity_label(rna, index)
                _copy_resource(
                    rna,
                    "unpairedMsaPath",
                    msa_dir / f"{prepared['name']}__{label}_unpairedmsa.a3m",
                    job_dir,
                )

    prepared_path = job_dir / f"{prepared['name']}_data.json"
    with prepared_path.open("w", encoding="utf-8") as handle:
        json.dump([prepared], handle, indent=2)
    return str(prepared_path)
