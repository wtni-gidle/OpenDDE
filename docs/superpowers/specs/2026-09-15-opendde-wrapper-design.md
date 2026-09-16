# OpenDDE AF3-Pro-Style Wrapper Design

## Scope

This fork keeps the OpenDDE model, checkpoint, tokenizer, feature meanings, and
sampling mathematics unchanged. The wrapper changes only the input hand-off,
data/inference scheduling, explicit-template loading, and prediction layout.

The implementation should stay small. Reuse the existing job-name validation,
MSA service, template search and featurization, inference runner, and CIF/JSON
writers. Do not add a framework of validators or duplicate the model runner.

## Public command

The normal entry point remains `opendde pred` and gains the same two stage
switches used by the local Protenix wrapper:

```text
-D, --run_data_pipeline BOOLEAN   default: true
-P, --run_inference BOOLEAN       default: true
--max_template_date YYYY-MM-DD    default: 2021-09-30
```

The four combinations mean:

| `-D` | `-P` | Behaviour |
| --- | --- | --- |
| true | false | Prepare portable per-job bundles; do not construct the model runner. |
| false | true | Read prepared JSON and predict without MSA/template search or JSON rewriting. |
| true | true | Prepare every job, then construct one runner and predict the prepared jobs. |
| false | false | Raise a short usage error before creating output or a runner. |

`opendde prep` remains as a convenience spelling for data-only preparation and
uses the same bundle writer. Existing `msa` and `mt` diagnostic commands remain
available.

## Input contract

The raw input remains the native OpenDDE top-level job list. The JSON filename
has no naming semantics. Every job's `name` is the user-visible identity and
determines its directory and prepared JSON filename.

Protein entities keep native fields such as `sequence`, `count`, `id`,
`pairedMsaPath`, and `unpairedMsaPath`. An entity with `id: ["A", "B"]`
shares one entity-level paired/unpaired MSA pair, named with the first ID.

Relative resource paths in a source JSON are resolved relative to that JSON.
Paths in a prepared JSON are relative to the prepared JSON. This removes any
dependency on the launch working directory.

## Template contract

`proteinChain.templates` follows AlphaFold 3's three-state behaviour:

- field omitted or `null`: automatic search during the data stage;
- `[]`: deliberately use no templates;
- non-empty list: use exactly those explicit templates and do not apply
  `max_template_date` filtering.

An explicit item has the AF3 path-and-residue-map fields:

```json
{
  "mmcifPath": "msas/job__A_template_0.cif",
  "queryIndices": [0, 1, 2],
  "templateIndices": [5, 6, 7]
}
```

AF3 assumes that such an mmCIF contains one protein chain. OpenDDE accepts that
form unchanged. Prepared templates originating from a normal multi-chain PDB
mmCIF may additionally contain `chainId`; it records the chain selected by the
existing OpenDDE template-hit processor and avoids altering the downloaded CIF.
No `chainId` is needed for a one-protein-chain mmCIF.

Automatic search uses `max_template_date`; explicit templates bypass it. The
prepared bundle contains the selected mmCIF files and residue mappings, so
inference-only does not need HMMER, Kalign, a template database, or PDBe.

## Prepared bundle

Data preparation produces one job per JSON:

```text
<out>/<name>/
├── <name>_data.json
└── msas/
    ├── <name>__A_unpairedmsa.a3m
    ├── <name>__A_pairedmsa.a3m
    ├── <name>__A_template_0.cif
    └── <name>__A_template_1.cif
```

Plain A3M/mmCIF files are the first implementation's canonical form. They are
easy to inspect and replace and require no new compression dependency.

The final `_data.json` is the only generated JSON. The wrapper must not create
`.opendde_preprocessed`, `*-update-msa.json`, or `*-final-updated.json`.
Search scratch data may live in a process-private temporary directory and is
deleted before the command returns.

Existing paired/unpaired resources are copied into the bundle. Automatically
searched resources are materialised with the same names. Existing explicit
template mmCIFs are copied into the bundle. Unknown JSON fields and unrelated
entity fields are preserved.

## Replacing unpaired MSA

The intended later workflow is to keep the prepared paired MSA and templates,
then replace only `unpairedMsaPath` or the file it references. Inference-only
reads the two paths independently and never searches.

The existing `msa_pair_as_unpair=true` behaviour remains unchanged: paired rows
are also merged into the unpaired pool and deduplicated. Cross-chain pairing
continues to come from the paired MSA.

## Prediction output

Prediction output is written directly in the AF3-Pro-style layout:

```text
<out>/<name>/
├── <name>_data.json
├── msas/
├── models/
│   └── seed-101_sample-0_model.cif
├── summary_confidences/
│   └── seed-101_sample-0_summary_confidences.json
└── full_data/
    └── seed-101_sample-0_full_data.json
```

The sample number is the model's original diffusion sample index, not a
confidence rank. Including the seed in every filename prevents collisions
without another `seed_*` directory. `full_data/` is written only when detailed
confidence output is enabled.

No legacy prediction tree is written in parallel.

## Compatibility and exclusions

- Native OpenDDE JSON remains valid.
- Existing `templatesPath` A3M/HHR input remains supported as a legacy direct
  inference path, but the wrapper data stage finalises it into `templates`.
- Model code is not modified.
- `--skip true` provides only canonical per-job/seed/sample completeness checks;
  there is no input/model hashing, resume database, locking, stale-output
  cleanup, archive format matrix, sidecar schema, recursive project manager, or
  silent job-name sanitisation. Single-device all-complete checks run before
  model construction. Fold-CP checks only after its existing Gloo control group
  exists, with rank 0 broadcasting the schedule to keep all ranks in the same
  collective control flow.
- `--write_now` is accepted for AF3 Pro compatibility, but OpenDDE always writes
  each prediction synchronously and does not implement delayed caching.
- Unsafe or duplicate names continue to use the validation already present in
  OpenDDE; the wrapper adds no second name-validation subsystem.
