# OpenDDE wrapper Lighthouse acceptance — 2026-09-16

## Verdict

The `feat/af3-pro-wrapper` checkout passed the planned Lighthouse installation,
portable-input, single-A40 inference, and wrapper/native equivalence acceptance.
For the fixed offline protein-only case, the wrapper and native OpenDDE revisions
produced exactly equal coordinates and confidence data. The wrapper-specific input
and output paths also passed compression, MSA-replacement, skip, JSON/NPZ, and
explicit-template integration checks.

This is a controlled regression result, not a claim that every possible input is
numerically invariant. The explicit-template run used only one cycle and one
diffusion step; its coordinates are an integration artifact and were not assessed
for scientific quality.

Acceptance root:

```text
/nfs/amino-projects/zhengwei/nwt/opendde_wrapper_acceptance_20260916
```

## Pinned sources and runtime

| Item | Accepted value |
| --- | --- |
| Wrapper revision | `3975e45851e4aee63bd675d98b7819ae5befc2d0` |
| Native revision | `ddfa1df8aff1babf1fddac4247b7d2351bd0ce9f` |
| Wrapper/native worktrees | Clean after testing |
| Environment | `$ROOT/venv`; Python 3.11; PyTorch `2.7.1+cu126` |
| Checkpoint | `$ROOT/runtime/checkpoint/opendde.pt` |
| Checkpoint SHA-256 | `7b826620390afad877ee2babc6a4d0df81b94d3a0be030959853d6a7da0807cc` |
| Install script SHA-256 | `5d2d5b5c908a13ecbee9af86426961e3e90222a5202094d76878f901ea12a5ef` |

The official checkpoint passed the repository manifest verifier at
2,625,249,069 bytes. The four common runtime assets were installed and separately
hashed under `$ROOT/provenance/runtime_sha256_size.tsv`. Search databases were
intentionally not downloaded.

## Slurm runs

| Job | Purpose | Result |
| --- | --- | --- |
| `27716780` | CPU installation and runtime verification | `COMPLETED 0:0`, `lh0400`, 4 CPU, 16 GiB, elapsed `00:14:02`, batch MaxRSS 8,490,360 KiB |
| `27717416` | First A40 attempt | Cancelled after `00:02:01`; excluded from acceptance |
| `27717654` | Corrected A40 acceptance | `COMPLETED 0:0`, `lh0471`, 8 CPU, 80 GiB, one NVIDIA A40, elapsed `00:05:20`, batch MaxRSS 4,049,080 KiB |

All three used the authorized `sigbio_project27` account; installation used
`sigbio`, and GPU testing used `sigbio_gpu`. The accepted GPU script exported by
Slurm has SHA-256
`38739318d1cb97aa32fdf7dae455e98c7162b6e7e781a53a39047956dee29bd0`.
The cancelled script has SHA-256
`046873056c0b9979012b838f4a80102580896829d561d42a3967257ef0c43970`.
Both job-specific hash records pass `sha256sum -c`.

Job `27717416` was cancelled when its native command resolved relative MSA paths
from the wrong working directory and began an online MSA refresh (`N_msa 9`). The
corrected job ran native inference from the prepared-bundle directory, kept the
same JSON, MSA bytes, checkpoint, seed, and model settings, and confirmed that no
MSA update was needed (`N_msa 2`). Partial output from the cancelled attempt is
retained only as diagnostic evidence.

## Fixed equivalence configuration

The native and wrapper production-schedule legs both used seed 101, one sample,
10 cycles, 200 diffusion steps, FP32, deterministic algorithms, TF32 disabled,
Torch triangle multiplication and attention kernels, fusion disabled, cache
enabled, protein MSA enabled, templates and RNA MSA disabled, and atom confidence
enabled. `CUDA_VISIBLE_DEVICES=0` selected the allocated A40, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` was set.

Both fresh processes imported `runner.batch_inference` from their intended pinned
checkout. The wrapper plain run also exercised the repository-root
`run_opendde.sh`.

## Native versus wrapper result

The comparison used no coordinate alignment and required exact atom identity and
order:

| Measurement | Observed | Acceptance gate |
| --- | ---: | ---: |
| Atoms compared | 250 | exact identity/order |
| Coordinate max absolute difference | `0 Å` | `<= 1e-3 Å` |
| Coordinate RMSD | `0 Å` | `<= 1e-3 Å` |
| Summary-confidence leaves | 15 | matching structure |
| Summary-confidence max difference | `0` | `<= 1e-5` |
| Full-confidence leaves | 3,081 | matching structure |
| Full-confidence max difference | `0` | `<= 1e-2` |

Primary evidence:

```text
$ROOT/logs/task3-native-wrapper-compare-27717654.log
$ROOT/provenance/task3-final-verification-27717654.txt
$ROOT/provenance/task3-{native-plain,wrapper-plain}-27717654.manifest
```

Accepted canonical outputs are rooted at:

```text
$ROOT/out-task3-native-plain-27717654/
$ROOT/out-task3-wrapper-plain-27717654/wrapper_accept/
```

## Wrapper workflow results

### Portable D-only bundles and compression

Data-only preparation produced exactly one `<name>_data.json` plus paired and
unpaired A3Ms under `msas/`, with no prediction tree and no hidden legacy
preprocessing files. The plain resources were byte-equal to their sources; the
`.zst` resources had zstd magic and decompressed byte-equal to the same sources.
After normalizing only the `.zst` suffixes, the prepared JSON documents were
semantically equal. Fresh execution of the preserved Task 2 verifier ended with
`ALL_TASK2_ASSERTIONS_PASSED`, and every recorded Task 2 hash passed.

The production wrapper outputs from plain and zstd prepared inputs were
byte-identical:

| Output | SHA-256 in both runs |
| --- | --- |
| Model CIF | `59369a171651befcead32b0bf6903e72c11ec0ef32083d2b9f23a0c7d1be4165` |
| Summary JSON | `8a755da115a2ca919b6ddce273ec61a4a93a93bb78ee5622d3c4c659f0c9f854` |
| Full-confidence JSON | `cdbc0ef811e6ed8151285d10812ca0eb6ad3c53720321e683b87c0212d47067f` |

### Manual unpaired-MSA replacement

Only the copied unpaired MSA was replaced. The prepared JSON and paired MSA kept
their original hashes, while plain A3M bytes placed under the existing `.a3m.zst`
name were correctly detected by content magic. Inference succeeded with
`N_msa 3`. Evidence and output are under:

```text
$ROOT/provenance/task3-replacement-*27717654*
$ROOT/logs/task3-replacement-27717654.log
$ROOT/out-task3-replacement-27717654/wrapper_accept/
```

### Skip and full-confidence format

For both JSON and NPZ full-confidence modes, `--skip true` left complete output
manifests unchanged and returned before environment, model, or checkpoint
initialization. Recomputing with `--compress_full_confidence true` removed the
stale JSON, wrote a valid NPZ, and matched all seven JSON values with maximum
absolute difference `0` (gate `<= 1e-7`).

### Explicit template

A separate offline smoke supplied `repo/examples/2lwu.cif` through AF3-like
`mmcifPath`, `queryIndices`, and `templateIndices`. D-only preparation preserved
the zero-based `0..14` mappings, copied the template to a portable `.cif.zst`, and
did not initialize the model. The prepared feature probe reported
`online_template_featurizer=None`, finite nonzero template features, and a
template atom-mask sum of 360. P-only inference with `use_template=true`, one
cycle, and one diffusion step produced parseable finite CIF, summary, and full
confidence files under the JSON `name` directory:

```text
$ROOT/template-smoke/predictions/template_smoke/
```

This verifies explicit-template path rewriting, offline reading, feature
construction, and model-path integration. It does **not** scientifically validate
the reduced-schedule coordinates or establish native/wrapper template-result
equivalence.

## Evidence locations

```text
$ROOT/jobs/install.sbatch
$ROOT/provenance/task2-commands.sh
$ROOT/provenance/task2-verify.py
$ROOT/provenance/task3-job-27717654-accepted-exact.sbatch
$ROOT/provenance/task3-commands-27717654.txt
$ROOT/provenance/task3-environment-27717654.txt
$ROOT/provenance/verify_task3_acceptance.sh
$ROOT/provenance/task3-final-verification-27717654.txt
$ROOT/logs/install-27716780.{out,err}
$ROOT/logs/task3-27717654.{out,err}
```

The final read-only verification on 2026-09-16 independently reran the Task 2 and
Task 3 gates, rechecked Slurm accounting, source revisions and cleanliness,
checkpoint and submitted-script hashes, comparison metrics, compression identity,
replacement protection, skip manifests, JSON/NPZ equality, explicit-template
evidence, and queue holds. It passed.

## Queue state and scope limits

Before the GPU run, the user explicitly authorized holding other pending jobs.
The preserved records account for 183 initially held jobs, one additional pending
job, and 11 later arrivals: 195 unique jobs. The final read-only check still found
exactly 195 pending jobs in `JobHeldUser`. No running job was held or cancelled,
and this reporting step did not change the hold state. They remain held pending
user direction.

Not exercised by this acceptance:

- automatic template search;
- external MSA search;
- RNA or RNA-MSA;
- multichain MSA pairing behavior;
- distributed Fold-CP.

`docs/inference_instructions.md` was reviewed and updated. It
documents name-driven output paths, relative MSA/template resolution, portable
D-only/P-only usage, compression and magic-byte behavior, manual unpaired-MSA
replacement, explicit-template cutoff behavior, `--skip`, synchronous output,
JSON/NPZ confidence, and Fold-CP. The cancelled native revision's CWD-sensitive relative
path behavior is not part of the wrapper's user-facing contract and does not expose
a wrapper documentation gap.
