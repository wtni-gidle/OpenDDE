# OpenDDE-Preview


![OpenDDE banner](https://raw.githubusercontent.com/aurekaresearch/OpenDDE/main/assets/OpenDDE.png)

![Status](https://img.shields.io/badge/status-preview-orange)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

OpenDDE is an open-source, all-atom biomolecular foundation model that turns co-folding into a scalable engine for structure prediction, design, and optimization in drug discovery.

> [!IMPORTANT]
> OpenDDE is a preview release. CLI flags, input/output JSON fields, and released
> checkpoints may change between versions, and predictions are not guaranteed to
> be reproducible across releases. It is not yet intended for production
> pipelines. Please open an issue for bugs, regressions, or feature requests.

![OpenDDE banner](https://raw.githubusercontent.com/aurekaresearch/OpenDDE/main/assets/scaling_law.png)
![results](https://raw.githubusercontent.com/aurekaresearch/OpenDDE/main/assets/results.png)

## News

- **2026-07-03: We release OpenDDE-preview (co-folding)! Read the [technical report](https://arxiv.org/abs/2607.03787) and visit the [website](https://aurekaresearch.github.io/OpenDDE-Website).**
    - Model weights can be downloaded from Hugging Face: [opendde.pt](https://huggingface.co/aurekaresearch/OpenDDE/resolve/eddd563ce96571f784012edd8f045181c8f8627d/opendde.pt) | [opendde_abag.pt](https://huggingface.co/aurekaresearch/OpenDDE/resolve/eddd563ce96571f784012edd8f045181c8f8627d/opendde_abag.pt)
    - The Docker image can be pulled with `docker pull aurekaresearch/opendde:v1`
    - The 2026ARK-AB Benchmark is now available

## Installation

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

See the [Docker guide](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/docker_installation.md)
for GPU setup, runtime-data mounts, and a complete `docker run` example.

> [!NOTE]
> `--torch-backend` selects the PyTorch build, while `[gpu]` adds the optional
> cuEquivariance kernels. Linux wheels require glibc 2.28 or newer. Apple
> Silicon runs on CPU or on the Metal (MPS) backend with `--device mps`; Intel
> macOS is unsupported, and Windows has not been validated. At runtime,
> `--device auto` uses CUDA when available, then MPS, and otherwise falls back
> to CPU.

For runtime-data setup and additional installation details, see the
[inference instructions](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/inference_instructions.md).

## Model and Runtime Data

OpenDDE reads checkpoints and runtime assets from `OPENDDE_ROOT_DIR`, defaulting
to `~/.cache/opendde` when the environment variable is unset:

```text
$OPENDDE_ROOT_DIR/
├── checkpoint/opendde.pt
├── common/
└── search_database/        # needed for local template/RNA-MSA preprocessing
```

From a source checkout, use the repository helper script to prepare the runtime
layout:

```bash
export OPENDDE_ROOT_DIR=/path/to/opendde_data
bash scripts/download_opendde_data.sh
```

The helper script lives in the source tree and is not installed with the
`opendde` Python package. If you installed OpenDDE from a wheel or package
index, either run the script from a cloned checkout, or let `opendde pred`
download the default checkpoint and common runtime files when they are missing.
You can also place checkpoint files manually under
`$OPENDDE_ROOT_DIR/checkpoint/`.

Python-managed checkpoint and common-asset downloads use a release-pinned
revision, published size, and SHA-256 before atomic replacement. The source
helper independently verifies released checkpoints; external search databases
are prepared separately. A checkpoint supplied with `--load_checkpoint_path`
is never replaced automatically.

For a prediction that disables protein MSA, template search, and RNA MSA, the
large `search_database/` files are not needed. From a source checkout, use:

```bash
bash scripts/download_opendde_data.sh --skip-search-database
```

Released checkpoints:

| Checkpoint        | Use case                              | Download                                                                                      |
| ----------------- | ------------------------------------- | --------------------------------------------------------------------------------------------- |
| `opendde.pt`      | General-purpose checkpoint.           | [opendde.pt](https://huggingface.co/aurekaresearch/OpenDDE/resolve/eddd563ce96571f784012edd8f045181c8f8627d/opendde.pt)           |
| `opendde_abag.pt` | Checkpoint tuned on antibody-antigen. | [opendde_abag.pt](https://huggingface.co/aurekaresearch/OpenDDE/resolve/eddd563ce96571f784012edd8f045181c8f8627d/opendde_abag.pt) |

Use `opendde.pt` with `-n opendde_v1` for the default model. For ABAG runs,
keep the filename as `opendde_abag.pt` and pass it explicitly:

```bash
opendde pred \
  -i input.json \
  -o ./output \
  --load_checkpoint_path "$OPENDDE_ROOT_DIR/checkpoint/opendde_abag.pt"
```

Detailed asset setup, mirrors, and Docker data mounts are documented in
[inference instructions](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/inference_instructions.md).

## Running Your First Prediction

Save this minimal OpenDDE input as `tiny.json`:

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

Run a small compatibility-oriented prediction. This disables external feature
searches, so only the checkpoint and common runtime files are required:

```bash
opendde pred \
  -i tiny.json \
  -o ./output \
  -n opendde_v1 \
  --use_msa false \
  --use_template false \
  --use_rna_msa false \
  --sample 1 \
  --step 200 \
  --cycle 10
```

Defaults are applied automatically: inference runs in `fp32`, triangle kernels
use `auto` dispatch, and seeds come from the job's `modelSeeds` unless `--seeds`
is provided. On CPU this example may be slow, but it avoids GPU-only kernels and
large search databases.

The data stage writes `output/tiny/tiny_data.json`. Predictions use the job's
`name`, regardless of the input JSON filename:

```text
output/tiny/
├── models/seed-101_sample-0_model.cif
├── summary_confidences/seed-101_sample-0_summary_confidences.json
└── full_data/seed-101_sample-0_full_data.npz
```

Sample numbers are original diffusion sample indices, not confidence ranks.
`full_data/` is written only with `--need_atom_confidence true` (the default).
Use `--skip true` to skip job/seed combinations whose required canonical files
are present and readable; incomplete or corrupt seeds are recomputed. The
prediction files are written synchronously before each job/seed completes.
Detailed confidence defaults to compressed NPZ; pass
`--compress_full_confidence false` to select JSON. Resume checks require the
selected format, and a successful rerun removes the stale JSON/NPZ alternate.

For production runs, enable the preprocessing features you need, for example
`--use_msa true`, `--use_template true`, or `--use_rna_msa true`. Those paths may
require network access, HMMER/Kalign binaries, and large local search databases;
see the inference guide for details.

## Prepare Once, Then Predict

`pred` runs both stages by default. To prepare portable inputs separately:

```bash
# Data only: enable the searches needed for your input
opendde pred -i input.json -o ./output -D true -P false \
  --use_template true

# Inference only: use the prepared job named "my_job"
opendde pred -i ./output/my_job/my_job_data.json -o ./output \
  -D false -P true --use_template true

# Both stages (the default)
opendde pred -i input.json -o ./output --use_template true

# Data-only convenience: protein MSA, templates, and RNA MSA when present
opendde prep -i input.json -o ./output

# Equivalent convenience runner; OPENDDE_BIN selects your environment command
./run_opendde.sh -i input.json -o ./output -D true -P false
```

Use your own `input.json`; `my_job` above is its job's `name`. Data-only never
loads the model. Inference-only also accepts a directory and recursively finds
only `*_data.json` bundles; it runs no searches and does not rewrite inputs.
Both stage switches default to `true`; setting both to `false` is an error.

A prepared protein bundle with chain ID `A` contains:

```text
<out>/<name>/
├── <name>_data.json
└── msas/
    ├── <name>__A_pairedmsa.a3m
    ├── <name>__A_unpairedmsa.a3m
    └── <name>__A_template_0.cif
```

Those plain suffixes are produced with `--compress_fold_input false`. The
default is zstd-compressed `.a3m.zst` / `.cif.zst`. Readers inspect magic bytes,
so replacing an unpaired `.a3m.zst` in place with ordinary A3M text is allowed;
the suffix alone does not force decompression.

Only supplied or generated MSA/template resources are included. Their paths are
relative to the JSON file, so the whole job directory can be moved. `FILE_`
ligands remain caller-managed external files, recorded with absolute paths in
the prepared JSON. You can replace the prepared
unpaired A3M in place and keep the paired A3M and templates for inference-only.
The existing `msa_pair_as_unpair=true` default also merges paired rows into the
unpaired pool and deduplicates them.

See the [native JSON example](examples/example_wrapper_input.json) and
[input format](docs/infer_json_format.md) for explicit templates and residue
mappings. The example's resource paths are illustrative; supply those files
before running it. Automatic templates use `--max_template_date` (default
`2021-09-30`); explicit templates bypass that cutoff.

## Multi-GPU Fold-CP Inference

> [!IMPORTANT]
> Multi-GPU Fold-CP does not support cuEquivariance triangle kernels, so use
> `--trimul_kernel torch --triatt_kernel torch`. Distributed `auto` requests
> resolve to these PyTorch kernels, while an explicit cuEquivariance request is
> rejected before model loading. On CUDA BF16, the distributed
> PyTorch triangle-attention path uses the Triton dependency from the GPU install
> extra to fuse attention-bias addition. See the
> [Fold-CP reproduction guide](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/foldcp_e2e_baseline.md)
> for controlled launch commands and validation guidance.

OpenDDE supports a `1 x P` Fold-CP inference mode for larger inputs, where `P`
can be any available GPU count greater than one. Prepare once in a single
process, then launch inference-only with one `torchrun` process per GPU.
Distributed data preparation is rejected. For example, four GPUs use:

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

Only the `1 x P` topology is supported. `--nproc_per_node` must equal
`--foldcp_size_cp P`; `--foldcp_size_dp` is retained only for command-line
compatibility and must remain `1`. The example uses `P=4`; replace both
occurrences of `4` with the desired `P`. For normal single-GPU or CPU inference,
omit the Fold-CP flags or use `--foldcp_mode single --foldcp_size_cp 1`.

## Input JSON

OpenDDE input is a top-level list of jobs. Each job contains `sequences` entries
such as `proteinChain`, `dnaSequence`, `rnaSequence`, `ligand`, or `ion`.
`covalent_bonds` is optional and should be added only when explicit covalent
links are needed.

See the [input JSON format](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/infer_json_format.md) for the full schema,
including covalent bonds, ligands, modifications, MSA paths, and template paths.

## CLI Overview

```bash
opendde pred    # prepare portable inputs and run inference; select stages with -D/-P
opendde doctor  # inspect Python/CUDA/kernel setup
opendde json    # convert PDB/CIF structures to OpenDDE JSON
opendde msa     # legacy low-level protein MSA preprocessing
opendde mt      # legacy low-level protein MSA + template preprocessing
opendde prep    # portable data-only bundles: protein MSA + templates + RNA MSA
```

Use `opendde <command> --help` for command-specific options. Public model names
currently include `opendde_v1`; use `--load_checkpoint_path` for alternate
checkpoint files such as `opendde_abag.pt`.

## Documentation

- [Changelog](https://github.com/aurekaresearch/OpenDDE/blob/main/CHANGELOG.md)
- [Model/checkpoint manifest](https://github.com/aurekaresearch/OpenDDE/blob/main/opendde/config/model_manifest.json)
- [Inference instructions](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/inference_instructions.md)
- [Docker installation](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/docker_installation.md)
- [Input JSON format](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/infer_json_format.md)
- [MSA/template/RNA-MSA pipeline](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/msa_template_pipeline.md)
- [Kernel options](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/kernels.md)
- [Fold-CP reproduction guide](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/foldcp_e2e_baseline.md)
- [Supported models](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/supported_models.md)
- [Tutorial](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/tutorial.md)

## Citation and Acknowledgements

If you use OpenDDE in your work, please cite this software and the related work.
OpenDDE builds on ideas and components from the AlphaFold 3 ecosystem, including
AlphaFold 3, Protenix, OpenFold, and ColabFold.

## License

OpenDDE is released under the Apache-2.0 license. See [LICENSE](https://github.com/aurekaresearch/OpenDDE/blob/main/LICENSE).

## Partnership and Collaboration

![Hiring](https://raw.githubusercontent.com/aurekaresearch/OpenDDE/main/assets/hiring.png)
