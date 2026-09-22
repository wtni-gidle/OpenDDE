# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Prepare JSON inference jobs with portable local resource bundles."""

from __future__ import annotations

from copy import deepcopy
import json
import os
import shutil
from os import PathLike
from pathlib import Path
import tempfile
from typing import Any

from opendde.data.inference.input_validation import validate_inference_jobs
from opendde.utils.text_io import read_text, write_zstd_text_atomic
from opendde.utils.scratch import temporary_directory


def _resolve_path(path: str, json_path: Path) -> str:
    resource_path = Path(path)
    if resource_path.is_absolute():
        return str(resource_path)
    return str(json_path.parent / resource_path)


def _resolve_chain_paths(chain: dict[str, Any], json_path: Path) -> None:
    for field in ("pairedMsaPath", "unpairedMsaPath"):
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
            msa = protein.get("msa")
            if isinstance(msa, dict) and isinstance(
                msa.get("precomputed_msa_dir"), str
            ):
                if msa["precomputed_msa_dir"]:
                    msa["precomputed_msa_dir"] = _resolve_path(
                        msa["precomputed_msa_dir"], json_path
                    )
        rna = sequence.get("rnaSequence")
        if isinstance(rna, dict) and isinstance(rna.get("unpairedMsaPath"), str):
            rna["unpairedMsaPath"] = _resolve_path(rna["unpairedMsaPath"], json_path)
        ligand = sequence.get("ligand")
        if isinstance(ligand, dict):
            value = ligand.get("ligand")
            if isinstance(value, str) and value.startswith("FILE_"):
                ligand["ligand"] = "FILE_" + _resolve_path(value[5:], json_path)
    return resolved


def load_input_jobs(input_path: str) -> list[tuple[Path, dict[str, Any]]]:
    """Load and validate jobs, resolving supported relative paths from the JSON file."""
    json_path = Path(input_path).resolve()
    with json_path.open(encoding="utf-8") as handle:
        jobs = validate_inference_jobs(json.load(handle))
    return [(json_path, resolve_job_paths(job, json_path)) for job in jobs]


def _sequential_entity_label(index: int) -> str:
    label = ""
    while True:
        index, remainder = divmod(index, 26)
        label = chr(ord("A") + remainder) + label
        if index == 0:
            return label
        index -= 1


def _entity_label(chain: dict[str, Any], used_labels: set[str]) -> str:
    entity_ids = chain.get("id")
    if isinstance(entity_ids, list) and entity_ids:
        label = str(entity_ids[0])
        from opendde.data.inference.input_validation import validate_sample_name

        validate_sample_name(label)
        if label.casefold() not in used_labels:
            used_labels.add(label.casefold())
            return label
    index = 0
    while True:
        label = _sequential_entity_label(index)
        if label.casefold() not in used_labels:
            used_labels.add(label.casefold())
            return label
        index += 1


def _copy_resource(
    chain: dict[str, Any],
    field: str,
    destination: Path,
    job_dir: Path,
    *,
    compress: bool,
) -> None:
    source = chain.get(field)
    inline = chain.get(field.removesuffix("Path"))
    if source is None and inline is None:
        return
    contents = inline if inline is not None else read_text(source)
    if compress:
        write_zstd_text_atomic(destination, contents)
    else:
        destination.write_text(contents, encoding="utf-8")
    chain[field] = destination.relative_to(job_dir).as_posix()
    chain.pop(field.removesuffix("Path"), None)


def _stage_prepared_job(
    job: dict[str, Any],
    out_dir: str | PathLike[str],
    *,
    compress_fold_input: bool = True,
) -> str:
    """Copy MSA/template resources and write JSON; FILE_ ligands stay external."""
    prepared = deepcopy(validate_inference_jobs([job])[0])
    job_dir = Path(out_dir) / prepared["name"]
    msa_dir = job_dir / "msas"
    msa_dir.mkdir(parents=True, exist_ok=True)
    msa_suffix = ".a3m.zst" if compress_fold_input else ".a3m"
    template_suffix = ".cif.zst" if compress_fold_input else ".cif"

    sequences = prepared.get("sequences")
    if isinstance(sequences, list):
        used_labels: set[str] = set()
        for sequence in sequences:
            if not isinstance(sequence, dict):
                continue
            protein = sequence.get("proteinChain")
            if isinstance(protein, dict):
                label = _entity_label(protein, used_labels)
                _copy_resource(
                    protein,
                    "pairedMsaPath",
                    msa_dir / f"{prepared['name']}__{label}_pairedmsa{msa_suffix}",
                    job_dir,
                    compress=compress_fold_input,
                )
                _copy_resource(
                    protein,
                    "unpairedMsaPath",
                    msa_dir / f"{prepared['name']}__{label}_unpairedmsa{msa_suffix}",
                    job_dir,
                    compress=compress_fold_input,
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
                            / f"{prepared['name']}__{label}_template_{template_index}{template_suffix}",
                            job_dir,
                            compress=compress_fold_input,
                        )
            rna = sequence.get("rnaSequence")
            if isinstance(rna, dict):
                label = _entity_label(rna, used_labels)
                _copy_resource(
                    rna,
                    "unpairedMsaPath",
                    msa_dir / f"{prepared['name']}__{label}_unpairedmsa{msa_suffix}",
                    job_dir,
                    compress=compress_fold_input,
                )

    prepared_path = job_dir / f"{prepared['name']}_data.json"
    with prepared_path.open("w", encoding="utf-8") as handle:
        json.dump([prepared], handle, indent=2)
    return str(prepared_path)


def _atomic_copy(source: Path, destination: Path):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, delete=False, prefix=f".{destination.name}."
        ) as handle:
            temporary = Path(handle.name)
            with source.open("rb") as reader:
                shutil.copyfileobj(reader, handle)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_prepared_job(job, out_dir, *, compress_fold_input=True):
    """Stage all reads first; publish JSON last and roll back caught write errors.

    This is not a multi-file crash transaction or a concurrent-writer protocol.
    """
    with temporary_directory(prefix="opendde-publish-") as scratch:
        staging = Path(scratch) / "staged"
        staged_json = Path(
            _stage_prepared_job(job, staging, compress_fold_input=compress_fold_input)
        )
        # The caller may intentionally select an alias (e.g. macOS /var or /tmp).
        # Resolve this trusted root, but still reject aliases inside the bundle.
        destination_root = Path(out_dir).resolve()
        files = sorted(
            p for p in staged_json.parent.rglob("*") if p.is_file() and p != staged_json
        )
        files.append(staged_json)
        backups, published, created = {}, [], []
        try:
            # Keep the established empty msas/ directory even for ligand-only jobs.
            msa_dir = (
                destination_root / staged_json.parent.relative_to(staging) / "msas"
            )
            missing = []
            for component in (msa_dir, *msa_dir.parents):
                if component.is_symlink():
                    raise ValueError(
                        f"Prepared output must not traverse a symlink: {component}"
                    )
                if not component.exists():
                    missing.append(component)
            for directory in reversed(missing):
                directory.mkdir()
                created.append(directory)
            for index, source in enumerate(files):
                destination = destination_root / source.relative_to(staging)
                for component in (destination, *destination.parents):
                    if component.is_symlink():
                        raise ValueError(
                            f"Prepared output must not traverse a symlink: {component}"
                        )
                missing = []
                parent = destination.parent
                while not parent.exists():
                    missing.append(parent)
                    parent = parent.parent
                for parent in reversed(missing):
                    parent.mkdir()
                    created.append(parent)
                if destination.exists():
                    backup = Path(scratch) / f"backup-{index}"
                    shutil.copyfile(destination, backup)
                    backups[destination] = backup
                _atomic_copy(source, destination)
                published.append(destination)
        except BaseException:
            for destination in reversed(published):
                if destination in backups:
                    _atomic_copy(backups[destination], destination)
                else:
                    destination.unlink(missing_ok=True)
            for directory in reversed(created):
                directory.rmdir()
            raise
        return str(Path(out_dir) / staged_json.relative_to(staging))


def prepare_input_jobs(
    input_path: str,
    out_dir: str,
    *,
    use_msa: bool = True,
    use_template: bool = False,
    use_rna_msa: bool = False,
    msa_server_mode: str | None = None,
    hmmsearch_binary_path: str | None = None,
    hmmbuild_binary_path: str | None = None,
    seqres_database_path: str | None = None,
    kalign_binary_path: str | None = None,
    nhmmer_binary_path: str | None = None,
    hmmalign_binary_path: str | None = None,
    hmmbuild_rna_binary_path: str | None = None,
    ntrna_database_path: str | None = None,
    rfam_database_path: str | None = None,
    rna_central_database_path: str | None = None,
    nhmmer_n_cpu: int | None = None,
    max_template_date: str = "2021-09-30",
    template_featurizer: Any = None,
    compress_fold_input: bool = True,
) -> list[str]:
    """Run searches in memory and publish one portable bundle per input job."""
    from opendde.config.data import data_configs
    from runner.msa_search import (
        convert_one_json_dict,
        need_msa_search,
        update_seq_msa,
    )
    from runner.rna_msa_search import update_rna_msa_info
    from runner.template_search import TemplateHitFeaturizer, update_template_info
    from opendde.data.template.template_finalizer import (
        reject_cached_template_featurizer,
        validate_explicit_template,
    )

    reject_cached_template_featurizer(template_featurizer)
    loaded_jobs = load_input_jobs(input_path)
    prepared_paths = []
    with temporary_directory(prefix="opendde-data-") as scratch_dir:
        for source_path, loaded_job in loaded_jobs:
            job = deepcopy(loaded_job)
            for sequence in job.get("sequences", []):
                chain = sequence.get("proteinChain", {})
                for template in chain.get("templates") or []:
                    validate_explicit_template(
                        chain["sequence"], template, base_dir=source_path.parent
                    )
            scratch = Path(scratch_dir) / job["name"]
            # Normalizing existing resources is independent of enabling search.
            job, _ = convert_one_json_dict(job)
            job = resolve_job_paths(job, source_path)
            if use_msa:
                if need_msa_search(job):
                    update_seq_msa(job, str(scratch / "msa"), mode=msa_server_mode)

            automatic_chains = (
                [
                    sequence["proteinChain"]
                    for sequence in job.get("sequences", [])
                    if "proteinChain" in sequence
                    and sequence["proteinChain"].get("templates") is None
                ]
                if use_template
                else []
            )
            if automatic_chains:
                # The search/finalizer writes next to its source MSA or hits.
                # Stage those sources so user inputs remain read-only.
                saved_conditions = []
                fields = (
                    "pairedMsa",
                    "pairedMsaPath",
                    "unpairedMsa",
                    "unpairedMsaPath",
                )
                for index, chain in enumerate(automatic_chains):
                    saved_conditions.append(
                        (chain, {k: chain[k] for k in fields if k in chain})
                    )
                    chain_dir = scratch / "templates" / str(index)
                    chain_dir.mkdir(parents=True, exist_ok=True)
                    for field in ("pairedMsaPath", "unpairedMsaPath"):
                        source = chain.get(field)
                        inline = chain.get(field.removesuffix("Path"))
                        if source is not None or inline is not None:
                            destination = chain_dir / f"{field}.a3m"
                            destination.write_text(
                                inline if inline is not None else read_text(source),
                                encoding="utf-8",
                            )
                            chain[field] = str(destination)
                            chain.pop(field.removesuffix("Path"), None)
                if template_featurizer is None:
                    template_config = data_configs["template"]
                    template_featurizer = TemplateHitFeaturizer(
                        mmcif_dir=template_config["prot_template_mmcif_dir"],
                        template_cache_dir=None,
                        max_hits=4,
                        kalign_binary_path=kalign_binary_path
                        or template_config["kalign_binary_path"],
                        release_dates_path=template_config["release_dates_path"],
                        obsolete_pdbs_path=template_config["obsolete_pdbs_path"],
                        _max_template_candidates_num=20,
                        fetch_remote=template_config["fetch_remote"],
                    )
                update_template_info(
                    [job],
                    hmmsearch_binary_path=hmmsearch_binary_path,
                    hmmbuild_binary_path=hmmbuild_binary_path,
                    seqres_database_path=seqres_database_path,
                    template_featurizer=template_featurizer,
                    max_template_date=max_template_date,
                )
                for chain, conditions in saved_conditions:
                    for field in fields:
                        chain.pop(field, None)
                    chain.update(conditions)
            if use_rna_msa:
                update_rna_msa_info(
                    [job],
                    out_dir=scratch_dir,
                    nhmmer_binary_path=nhmmer_binary_path,
                    hmmalign_binary_path=hmmalign_binary_path,
                    hmmbuild_binary_path=hmmbuild_rna_binary_path
                    or hmmbuild_binary_path,
                    ntrna_database_path=ntrna_database_path,
                    rfam_database_path=rfam_database_path,
                    rna_central_database_path=rna_central_database_path,
                    nhmmer_n_cpu=nhmmer_n_cpu,
                )
            prepared_paths.append(
                write_prepared_job(
                    job,
                    out_dir,
                    compress_fold_input=compress_fold_input,
                )
            )
    return prepared_paths
