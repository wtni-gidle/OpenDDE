# Inference Instructions

Concise reference for installing OpenDDE, preparing runtime data, and running
`opendde` commands.

## Install

OpenDDE supports CPython `3.11`, `3.12`, and `3.13`. We recommend
[`uv`](https://docs.astral.sh/uv/getting-started/installation/) for Python
installations. Choose one of the following methods.

### Install from PyPI

```bash
uv venv --python 3.11
```

CPU:

```bash
uv pip install --python .venv --torch-backend cpu opendde
```

NVIDIA GPU (Linux x86_64, CUDA 12.6):

```bash
uv pip install --python .venv --torch-backend cu126 "opendde[gpu]"
```

### Install from source

```bash
git clone https://github.com/aurekaresearch/OpenDDE.git
cd OpenDDE
uv venv --python 3.11
```

CPU:

```bash
uv pip install --python .venv --torch-backend cpu -e .
```

NVIDIA GPU (Linux x86_64, CUDA 12.6):

```bash
uv pip install --python .venv --torch-backend cu126 -e ".[gpu]"
```

After a PyPI or source installation, verify the environment with:

```bash
uv run --no-project --python .venv opendde doctor
```

### Use Docker

The prebuilt image targets NVIDIA GPU inference:

```bash
docker pull aurekaresearch/opendde:v1
```

See the [Docker guide](./docker_installation.md) for GPU setup, runtime-data
mounts, and a complete `docker run` example.

> [!NOTE]
> `--torch-backend` selects the PyTorch build, while `[gpu]` adds the optional
> cuEquivariance kernels. Linux wheels require glibc 2.28 or newer. Apple
> Silicon runs on CPU or on the Metal (MPS) backend with `--device mps`; Intel
> macOS is unsupported, and Windows has not been validated. At runtime,
> `--device auto` uses CUDA when available, then MPS, and otherwise falls back
> to CPU.

## Runtime data

Set `OPENDDE_ROOT_DIR` to the directory that stores checkpoints and runtime data:

```text
$OPENDDE_ROOT_DIR/
├── checkpoint/opendde.pt
├── common/
└── search_database/        # needed for local template/RNA-MSA search
```

The default checkpoint and managed common files come from a release-pinned
asset revision and are verified against their published size and SHA-256.
Checkpoints passed explicitly with `--load_checkpoint_path` are left untouched.

Prepare data from a source checkout:

```bash
export OPENDDE_ROOT_DIR=/path/to/opendde_data
bash scripts/download_opendde_data.sh
```

For a protein-only prediction that disables MSA, template, and RNA-MSA features,
search databases are not needed:

```bash
bash scripts/download_opendde_data.sh --skip-search-database
```

If you already have a custom checkpoint, keep a descriptive filename and pass
it directly. Use `--skip-model` when preparing only the remaining runtime data:

```bash
mkdir -p "$OPENDDE_ROOT_DIR/checkpoint"
cp /path/to/my_checkpoint.pt \
  "$OPENDDE_ROOT_DIR/checkpoint/my_checkpoint.pt"
bash scripts/download_opendde_data.sh --skip-model
opendde pred \
  --load_checkpoint_path "$OPENDDE_ROOT_DIR/checkpoint/my_checkpoint.pt" \
  -i examples/input.json \
  -o ./output
```

The names `opendde.pt` and `opendde_abag.pt` are reserved for released assets.
Their authoritative links, sizes, and digests are in
[supported_models.md](./supported_models.md).

Use `opendde.pt` with `-n opendde_v1` as the default general-purpose
checkpoint. To use the ABAG-optimized checkpoint, keep it as
`opendde_abag.pt` and pass it with `--load_checkpoint_path`, for example
`opendde pred --load_checkpoint_path "$OPENDDE_ROOT_DIR/checkpoint/opendde_abag.pt"`.

Install and verify the ABAG checkpoint from the same manifest-backed helper:

```bash
export OPENDDE_ROOT_DIR=/path/to/opendde_data
bash scripts/download_opendde_data.sh \
  --checkpoint opendde_abag.pt \
  --skip-common \
  --skip-search-database
```

Then run general-purpose inference without an explicit checkpoint path. For ABAG
inference, add:

```bash
--load_checkpoint_path "$OPENDDE_ROOT_DIR/checkpoint/opendde_abag.pt"
```

Useful environment variables:

| Variable | Purpose |
| --- | --- |
| `OPENDDE_ROOT_DIR` | Checkpoints, common files, search databases. Defaults to `~/.cache/opendde`. |
| `OPENDDE_DEPENDENCY_URL` | Override checkpoint download root. |
| `OPENDDE_COMMON_URL` | Override common runtime file download root. Falls back to `OPENDDE_DEPENDENCY_URL` when set. |
| `OPENDDE_SEARCH_DATABASE_URL` | Override template/RNA-MSA database download root. |
| `LAYERNORM_TYPE` | LayerNorm backend; defaults to `torch`. Set to `fast_layernorm` to opt into the fused kernel. |

Automatic template/RNA-MSA preparation also needs HMMER. Template preparation
may need `kalign` for realignment; inference from explicit prepared templates
needs neither tool:

```bash
apt-get update && apt-get install -y hmmer kalign
```

## Input JSON

OpenDDE input is a top-level list of jobs:

```json
[
  {
    "name": "tiny",
    "modelSeeds": [101],
    "sequences": [
      {
        "proteinChain": {
          "sequence": "ACDEFGHIK",
          "count": 1
        }
      }
    ]
  }
]
```

`covalent_bonds` is optional and may be omitted from a job; include it only to
declare explicit covalent links between entities.

Full schema: [infer_json_format.md](./infer_json_format.md).

Each job's `name` sets its output directory, independently of the source JSON
filename. Protein MSA/template paths and RNA MSA paths resolve relative to the
JSON file. The [wrapper example](../examples/example_wrapper_input.json) shows
chain IDs and explicit template mappings; its A3M/mmCIF resource files are
illustrative and must be supplied before running it.

Convert a structure file to JSON:

```bash
opendde json -i examples/7pzb.pdb -o ./output --altloc first
opendde json -i examples/2lwu.cif -o ./output --altloc first --assembly_id 1
```

## Prepare portable inputs

For an input file `input.json` containing a job named `my_job`:

```bash
# Data only: enable the features needed by your input
opendde pred -i input.json -o ./output -D true -P false \
  --use_template true --use_rna_msa true

# Data-only convenience, enabling protein MSA, templates, and RNA MSA
opendde prep -i input.json -o ./output

# Inference only: consume the prepared bundle
opendde pred -i ./output/my_job/my_job_data.json -o ./output \
  -D false -P true --use_template true --use_rna_msa true
```

`-D/--run_data_pipeline` and `-P/--run_inference` are booleans, both defaulting
to `true`. Data-only never loads the model; both `false` is an error. `pred`
defaults to protein MSA enabled, templates and RNA MSA disabled. `prep` enables
all three where applicable and prints each prepared JSON path.

`-J/--write_input_json` independently controls public input JSON/resource writes.
When omitted it follows `-D`. With `-D true -J false`, preparation remains private
until inference finishes and is then cleaned up. With `-D false -J true`, supplied
conditions are saved without searching. Public templates must use the main JSON's
`templates` list; legacy `templatesPath` is rejected. See the
[MSA/template guide](msa_template_pipeline.md#independent-publication-and-temporary-files).

For a protein entity with ID `A`, preparation writes:

```text
<out>/<name>/
├── <name>_data.json
└── msas/
    ├── <name>__A_pairedmsa.a3m
    ├── <name>__A_unpairedmsa.a3m
    └── <name>__A_template_0.cif
```

The plain names above use `--compress_fold_input false`. By default `pred` and
`prep` append `.zst` and write standard zstd frames. Readers detect compression
from magic bytes, so a manually replaced plain A3M is valid even if its path
still ends in `.zst`.

Only supplied/generated MSA/template resources appear, including RNA MSA.
The single-job JSON refers to these copied resources with relative paths;
move the entire job directory together to preserve them. `FILE_` ligand files
remain caller-managed absolute external references. Keep those files accessible
or update their paths after moving to another machine. Search scratch files are
temporary; the data stage writes one final JSON per job.
Inference-only accepts the prepared JSON directly or recursively discovers only
`*_data.json` bundles in a directory. It runs no searches and does not rewrite
the input JSON. Keep the relevant `--use_*` flags enabled to consume features.

You can replace just the unpaired A3M file in place before inference-only,
keeping paired A3M and templates. The `msa_pair_as_unpair=true` default also
merges paired rows into the unpaired pool with deduplication; the paired input
still supplies cross-chain pairing. See the pipeline guide for supported
species identifiers in A3M headers.

Search behavior:

- Protein MSA uses the public ColabFold MMseqs2 API unless A3M paths are already
  present in the JSON.
- Template and RNA-MSA search use local databases under
  `$OPENDDE_ROOT_DIR/search_database/`.
- With `--use_template true`, omitted or `null` `templates` requests automatic
  data-stage selection; `[]` uses no templates; a non-empty list uses explicit
  mmCIFs and residue mappings. Automatic selection uses `--max_template_date`
  (default `2021-09-30`); explicit templates bypass the cutoff.
- Prepared explicit templates need no HMMER, Kalign, template database, or PDBe
  access during inference. Checkpoints and common runtime assets remain needed.

The low-level `msa`/`mt` diagnostic commands retain legacy intermediate files
and hit-file behavior. Details: [msa_template_pipeline.md](./msa_template_pipeline.md).

## Run prediction

Standard run (both data and inference stages):

```bash
opendde pred -i examples/input.json -o ./output -n opendde_v1
```

Compatibility run with the standard step/cycle counts:

```bash
opendde pred \
  -i examples/input.json \
  -o ./output \
  -n opendde_v1 \
  --use_msa false \
  --use_template false \
  --use_rna_msa false \
  --sample 1 \
  --step 200 \
  --cycle 10
```

Inference defaults to `--device auto`, `fp32`, and `auto` triangle kernels.
Device auto-selection uses NVIDIA CUDA when available, then the Apple Metal
(MPS) backend on Apple Silicon, and otherwise CPU.
cuEquivariance is selected only when its Linux CUDA packages import successfully;
otherwise the model uses PyTorch triangle kernels. MPS always uses PyTorch
triangle kernels and defaults to FP32. `--dtype bf16` also works there and
follows the same dynamic policy as CUDA: the trunk uses BF16; by default,
diffusion and the confidence head stay FP32 through 2560 tokens, confidence
uses BF16 above 2560, and diffusion uses BF16 above 3840 to reduce memory.
BF16 performance varies by Apple GPU and workload, so compare it with FP32 for
your inputs. BF16 autocast needs macOS 14 or newer and is downgraded to FP32
below that.

## Multi-GPU Fold-CP inference

> [!IMPORTANT]
> Fold-CP inference does not currently support cuEquivariance (`cueq`) triangle
> kernels. Select the distributed PyTorch implementations with
> `--triatt_kernel torch --trimul_kernel torch`. In distributed mode, `auto`
> resolves to these PyTorch kernels and an explicit `cueq` request fails before
> model loading. On CUDA BF16, Fold-CP triangle
> attention also uses Triton 3.3.1 from the GPU install extra to fuse
> attention-bias addition; this Triton helper is separate from cuEquivariance.

Fold-CP distributes token-pair-heavy inference work over a `1 x P` mesh, where
`P` can be any available GPU count greater than one. Prepare inputs once in a
single process, then use `torchrun` with `-D false -P true`; distributed data
preparation is rejected. For example, four GPUs use:

```bash
opendde pred -i examples/protein_200.json -o ./output_foldcp \
  -D true -P false \
  --use_msa false --use_template false --use_rna_msa false

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node 4 \
  -m runner.batch_inference pred \
  -i ./output_foldcp/synthetic_protein_200/synthetic_protein_200_data.json \
  -o ./output_foldcp \
  -D false -P true \
  -n opendde_v1 \
  --use_msa false \
  --use_template false \
  --use_rna_msa false \
  --sample 1 \
  --step 200 \
  --cycle 10 \
  --trimul_kernel torch \
  --triatt_kernel torch \
  --foldcp_mode distributed \
  --foldcp_size_dp 1 \
  --foldcp_size_cp 4
```

Runtime notes:

- `--nproc_per_node P` must match `--foldcp_size_dp 1` times
  `--foldcp_size_cp P`.
- `--foldcp_size_cp P` creates a `1 x P` context-parallel mesh. The example
  uses `P=4`; other GPU counts require changing both `4` values.
- Keep the input, model, dtype, cycle, step, sample, MSA, and template settings
  identical when comparing single-GPU and Fold-CP outputs. Select
  `--triatt_kernel torch --trimul_kernel torch` for both runs; the distributed
  CUDA BF16 path additionally uses Triton for attention-bias fusion.
- Outputs are written under the requested `-o/--out_dir` just like normal
  inference.
- Optional `--foldcp_metrics_jsonl path/to/metrics.jsonl` records Fold-CP timing
  and memory metrics.

For single-GPU inference, omit the Fold-CP flags or set
`--foldcp_mode single --foldcp_size_cp 1`.

## Optional TFG Guidance

OpenDDE includes default-off Training-Free Guidance (TFG) for protein-ligand
runs. TFG refines each sampled trajectory with geometry potentials while keeping
the requested `--sample` count unchanged.

```bash
opendde pred -i examples/input.json -o ./output -n opendde_v1 \
  --use_tfg_guidance true
```

## Prediction outputs

Prediction writes directly under the requested `-o` directory:

```text
<out>/<name>/
├── models/seed-101_sample-0_model.cif
├── summary_confidences/seed-101_sample-0_summary_confidences.json
└── full_data/seed-101_sample-0_full_data.npz
```

Filenames use the original diffusion sample index, not confidence rank. Seeds
are part of every filename. `full_data/` is written only with
`--need_atom_confidence true` (the default). The prepared JSON and `msas/` remain
where the data stage wrote them, even when inference uses a different output
directory.

The detailed path defaults to `.npz`; use `--compress_full_confidence false`
to select JSON.
Each NPZ key is an existing OpenDDE full-confidence field stored as a primitive
NumPy array and can be loaded with `numpy.load(path, allow_pickle=False)`.

For lightweight resume, pass `--skip true`. OpenDDE checks each requested
job/seed and original sample index: the model CIF and summary confidence JSON
must exist as non-empty regular files, as must `full_data` in the selected JSON
or NPZ format when atom confidence is enabled. Resume checks file metadata only;
it does not open or validate output contents or compare inputs, model settings,
or other conditions. Existing outputs can therefore be reused after conditions
change. All expected sample indices must be present. If any required file is
missing or empty, the entire affected seed is recomputed with all its samples;
completed seeds are preserved. Existing job-name and path-containment checks
still apply. The
default `--skip false` always recomputes and does not inspect existing prediction
outputs; data-only runs never inspect them either.

Single-device inference can take the all-complete fast path before constructing
the model runner. Fold-CP first constructs its runner and Gloo control group;
rank 0 then checks completeness and broadcasts the selected schedule so every
rank collectively skips or runs the same work.

OpenDDE writes every job/seed synchronously; there is no delayed-write mode.

## Common flags

| Flag | Meaning |
| --- | --- |
| `-D`, `--run_data_pipeline` | Prepare portable bundles; boolean, default `true`. Use a single process for this stage. |
| `-P`, `--run_inference` | Predict from prepared inputs; boolean, default `true`. |
| `-J`, `--write_input_json` | Publish portable input JSON/resources; unset follows `-D`. False keeps preparation private. |
| `-n`, `--model_name` | Model name. Currently `opendde_v1`. |
| `--load_checkpoint_path` | Explicit checkpoint path. |
| `--seeds` | Comma-separated seeds, e.g. `101,102`. Overrides the job's `modelSeeds`; if unset, `modelSeeds` are used, or a random seed when both are absent. |
| `--use_msa` | Use/generate protein MSA features. |
| `--use_template` | Use/generate template features. |
| `--max_template_date` | Automatic data-stage template cutoff, default `2021-09-30`. Explicit templates bypass it. |
| `--use_rna_msa` | Use/generate RNA MSA features; requires `--use_msa true`. |
| `--need_atom_confidence` | Write detailed confidence in `full_data/`; default `true`. |
| `--compress_fold_input` | Write prepared MSA/template text as `.zst`; default `true`. Plain text manually placed under that suffix is still readable. |
| `--compress_full_confidence` | Write detailed confidence as compressed NPZ instead of JSON; default `true`. |
| `--skip` | Skip complete job/seed outputs; boolean, default `false`. |
| `--use_tfg_guidance` | Enable Training-Free Guidance. |
| `--foldcp_mode` | `single` or `distributed`; use `distributed` with `torchrun` for multi-GPU Fold-CP inference. |
| `--foldcp_size_dp` | Compatibility option; only `1` is supported. Runtime `2 x 2` topology is not maintained. |
| `--foldcp_size_cp` | Context-parallel degree `P` in the maintained `1 x P` mesh; must match the launched process count. |
| `--foldcp_devices` | Optional visible-device list recorded in Fold-CP metrics; actual GPU visibility is controlled by `CUDA_VISIBLE_DEVICES`. |
| `--foldcp_metrics_jsonl` | Optional JSONL path for Fold-CP timing and memory metrics. |
| `--dtype` | `bf16` or `fp32`. |
| `--device` | `auto`, `cpu`, `cuda`, or `mps`; auto uses CUDA when available, then Apple MPS, and otherwise CPU. |
| `--trimul_kernel`, `--triatt_kernel` | `auto`, `cuequivariance`, or `torch`. |

Run `opendde <command> --help` for the full option list.
