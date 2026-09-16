# OpenDDE Lighthouse Acceptance Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install the wrapped OpenDDE branch on Lighthouse, exercise its portable input/output workflow, and prove that wrapper-only changes do not alter native model predictions.

**Architecture:** Keep every generated artifact under `/nfs/amino-projects/zhengwei/nwt/opendde_wrapper_acceptance_20260916`. Build one Python 3.11 CUDA environment and use it for both native commit `ddfa1df8aff1babf1fddac4247b7d2351bd0ce9f` and wrapper commit `3975e45851e4aee63bd675d98b7819ae5befc2d0`; switch source code only with `PYTHONPATH`. Use a CPU Slurm job for installation/download and one A40 job for all deterministic inference comparisons.

**Tech Stack:** Slurm 25.11, an authorized SigBio account (`sigbio_project1` or `sigbio_project27`), Python 3.11, PyTorch 2.7.1/cu126, OpenDDE v1.1.1, NVIDIA A40.

**Spec:** `docs/superpowers/specs/2026-09-15-opendde-wrapper-design.md`

## Global Constraints

- Write only below `/nfs/amino-projects/zhengwei/nwt` on Lighthouse.
- Use an authorized SigBio account and record it; prefer `sigbio_project1`, but use `sigbio_project27` when current fairshare produces a materially earlier start. CPU work uses `sigbio`, GPU work uses `sigbio_gpu --gres=gpu:1`.
- Do not modify the shared AF3 environments, checkpoints, or databases.
- Pin native code to `ddfa1df8aff1babf1fddac4247b7d2351bd0ce9f` and wrapper code to `3975e45851e4aee63bd675d98b7819ae5befc2d0`.
- Hold checkpoint, prepared JSON, MSA bytes, seed `101`, model cycles/steps, dtype, kernels, and CUDA visibility constant in the native/wrapper A/B run.
- Do not compare an AF3 checkpoint with OpenDDE; download and verify OpenDDE's released `opendde.pt`.

---

### Task 1: Install and verify runtime assets

**Files:**
- Create remotely: `/nfs/amino-projects/zhengwei/nwt/opendde_wrapper_acceptance_20260916/jobs/install.sbatch`
- Create remotely: `/nfs/amino-projects/zhengwei/nwt/opendde_wrapper_acceptance_20260916/venv/`
- Create remotely: `/nfs/amino-projects/zhengwei/nwt/opendde_wrapper_acceptance_20260916/runtime/`

**Interfaces:**
- Consumes: wrapper checkout at the pinned revision.
- Produces: `venv/bin/opendde`, verified `runtime/checkpoint/opendde.pt`, common runtime files, and provenance logs.

- [x] **Step 1: Submit an isolated CPU installation job**

Use `sigbio`, initially 8 CPUs, 32 GiB RAM, and four hours; a 4-CPU, 16-GiB, two-hour backfill request is acceptable if the initial job has no start estimate. Load `python/3.11.5`, create the venv, install `torch==2.7.1` from the cu126 index, then install the checkout with `pip install -e '.[gpu]'`.

- [x] **Step 2: Download only required inference assets**

Run `scripts/download_opendde_data.sh --skip-search-database` with `OPENDDE_ROOT_DIR` set to the acceptance runtime and `OPENDDE_PYTHON` set to the venv Python.

- [x] **Step 3: Verify installation provenance**

Record `pip freeze`, `opendde doctor`, checkpoint SHA-256, Python/Torch/CUDA versions, repository revisions, and CLI help. Require successful exit codes and a clean wrapper checkout.

### Task 2: Build the offline fixture and native checkout

**Files:**
- Create remotely: `/nfs/amino-projects/zhengwei/nwt/opendde_wrapper_acceptance_20260916/native/`
- Create remotely: `/nfs/amino-projects/zhengwei/nwt/opendde_wrapper_acceptance_20260916/case/raw.json`
- Create remotely: paired, unpaired, and replacement A3M files below `case/assets/`.

**Interfaces:**
- Consumes: installed wrapper environment and the source repository history.
- Produces: one 29-residue single-chain offline job named `wrapper_accept`, with model seed `101` and local paired/unpaired MSAs.

- [x] **Step 1: Add the detached native worktree**

Run `git -C repo worktree add --detach native ddfa1df8aff1babf1fddac4247b7d2351bd0ce9f` and record its revision.

- [x] **Step 2: Write the fixture**

Use sequence `MKTAYIAKQRQISFVKSHFSRQDILDLWQ`; the original MSAs contain the query and one E. coli sequence, while the replacement unpaired MSA adds one same-species variant.

- [x] **Step 3: Produce compressed and plain D-only bundles**

Run wrapper D-only once with `--compress_fold_input true` and once with `false`. Check canonical paths, magic bytes/decompressed contents, semantic JSON equality after suffix normalization, and absence of prediction files or hidden legacy preprocessing outputs.

### Task 3: Run the controlled A/B inference and wrapper workflow tests

**Files:**
- Create remotely: `/nfs/amino-projects/zhengwei/nwt/opendde_wrapper_acceptance_20260916/jobs/acceptance_gpu.sbatch`
- Create remotely: outputs and logs below the acceptance root only.

**Interfaces:**
- Consumes: the same plain prepared JSON, environment, checkpoint, seed, and fixed model configuration for both code revisions.
- Produces: native and wrapper predictions plus compression, replacement, skip, and NPZ evidence.

- [x] **Step 1: Submit one A40 job**

Request `sigbio_gpu`, one GPU, 8 CPUs, 80 GiB RAM, and eight hours. Set `CUDA_VISIBLE_DEVICES=0` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`; use FP32, deterministic algorithms, TF32 off, torch triangle kernels, fusion off, seed 101, one sample, 10 cycles, and 200 diffusion steps.

- [x] **Step 2: Compare native and wrapper predictions**

Run native with `PYTHONPATH=native` and wrapper with `PYTHONPATH=repo` in fresh processes against the exact same uncompressed prepared JSON. Require exact atom identity/order, unaligned all-atom maximum and RMSD coordinate differences no greater than `1e-3` Å, summary numeric differences no greater than `1e-5`, and full-confidence numeric differences no greater than `1e-2`.

- [x] **Step 3: Verify fold-input compression invariance**

Run P-only on compressed and plain bundles with the wrapper and require byte-identical model CIF, summary JSON, and full-confidence JSON.

- [x] **Step 4: Verify manual unpaired-MSA replacement**

Overwrite only the `.a3m.zst` unpaired path with plain A3M text, retain the JSON and paired MSA hashes, and require successful P-only inference.

- [x] **Step 5: Verify skip and NPZ behavior**

Require JSON-mode skip to leave the output manifest unchanged. Switch to compressed NPZ, require stale JSON removal and elementwise equality to the saved JSON values, then require NPZ-mode skip to leave the manifest unchanged.

### Task 4: Review evidence and publish an acceptance report

**Files:**
- Create remotely: `/nfs/amino-projects/zhengwei/nwt/opendde_wrapper_acceptance_20260916/ACCEPTANCE_REPORT.md`
- Modify locally after verification: `docs/inference_instructions.md` only if the live run exposes an actual documentation gap.

**Interfaces:**
- Consumes: Slurm logs, provenance, output manifests, numerical comparison results, and exact job IDs.
- Produces: a pass/fail report with paths and limitations.

- [x] **Step 1: Check Slurm completion and logs**

Require `sacct` state `COMPLETED` and exit code `0:0` for installation and GPU jobs; inspect both stdout and stderr.

- [x] **Step 2: Re-run lightweight evidence checks from the login node**

Confirm revisions, checkpoint hash, expected file tree, zstd magic, JSON/NPZ readability, and clean source checkout without invoking the model.

- [x] **Step 3: Record limitations honestly**

State whether automatic template search, RNA MSA search, multi-GPU FoldCP, or multiple-job scheduling were excluded from this minimal acceptance gate.
