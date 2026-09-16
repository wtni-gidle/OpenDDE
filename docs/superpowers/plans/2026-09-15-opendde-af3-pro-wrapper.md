# OpenDDE AF3-Pro-Style Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thin, native-JSON OpenDDE workflow that prepares portable per-job inputs and writes AF3-Pro-style prediction outputs without changing the model.

**Architecture:** A focused `runner/fold_input.py` module owns JSON-relative paths and per-job bundle materialisation. Existing MSA/template search functions are called in memory with temporary scratch space; prediction reuses the existing runner with preprocessing disabled. Explicit mmCIF templates are converted directly into the same template feature dictionaries already assembled by OpenDDE.

**Tech Stack:** Python 3.11-3.13, Click, pytest, existing OpenDDE MSA/template/inference code.

**Spec:** `docs/superpowers/specs/2026-09-15-opendde-wrapper-design.md`

## Global Constraints

- Keep native OpenDDE JSON; `job["name"]` is the sole user-visible output identity.
- Do not create `.opendde_preprocessed`, `*-update-msa.json`, or `*-final-updated.json` in the wrapper workflow.
- Do not modify model, checkpoint, tokenizer, feature meanings, or sampling mathematics.
- Keep `msa_pair_as_unpair=true` semantics unchanged.
- Use plain `.a3m` and `.cif` prepared resources; add no compression dependency.
- Use existing OpenDDE name validation; do not build another validation framework.
- Use test-first red/green cycles for production changes and make task-focused commits.

---

### Task 1: JSON-relative paths and portable per-job bundle

**Files:**
- Create: `runner/fold_input.py`
- Create: `tests/test_fold_input_bundle.py`

**Interfaces:**
- Produces: `load_input_jobs(input_path: str) -> list[tuple[Path, dict[str, Any]]]`.
- Produces: `resolve_job_paths(job: dict[str, Any], json_path: Path) -> dict[str, Any]`.
- Produces: `write_prepared_job(job: dict[str, Any], out_dir: str | PathLike[str]) -> str`.
- `write_prepared_job` copies protein paired/unpaired MSA and explicit template mmCIF resources into `<out>/<name>/msas/`, then writes a single-item `<name>_data.json` whose resource paths are relative to that file.

- [ ] **Step 1: Write failing bundle tests**

Create fixtures with an arbitrarily named source JSON containing a job named
`target`, `id: ["A", "B"]`, paired/unpaired A3M files, an explicit template,
and an unknown field. Assert the prepared paths are exactly:

```python
job_dir = tmp_path / "out" / "target"
assert prepared == str(job_dir / "target_data.json")
assert (job_dir / "msas" / "target__A_pairedmsa.a3m").read_text() == paired
assert (job_dir / "msas" / "target__A_unpairedmsa.a3m").read_text() == unpaired
assert (job_dir / "msas" / "target__A_template_0.cif").read_text() == mmcif
assert loaded[0]["unknown"] == "preserved"
assert loaded[0]["sequences"][0]["proteinChain"]["pairedMsaPath"] == (
    "msas/target__A_pairedmsa.a3m"
)
```

Also assert relative source paths resolve from the source JSON directory, not
the test CWD, and two jobs produce two name-based directories.

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_fold_input_bundle.py -q`

Expected: collection fails because `runner.fold_input` does not exist.

- [ ] **Step 3: Implement the minimal bundle module**

Use `deepcopy`, `Path`, `shutil.copyfile`, `json.load/json.dump`, and the
existing `validate_inference_jobs`. Resolve only the path-bearing fields used
by this workflow: protein paired/unpaired MSA, protein `templatesPath`,
`templates[*].mmcifPath`, and RNA unpaired MSA. Choose the entity label from
the first `id`, falling back to sequential `A`, `B`, ... labels. Preserve all
other fields verbatim. Write directly to the final `_data.json`; do not create
another JSON name.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_fold_input_bundle.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add runner/fold_input.py tests/test_fold_input_bundle.py
git commit -m "feat: add portable OpenDDE input bundles"
```

### Task 2: AF3-style explicit template features and automatic finalisation

**Files:**
- Create: `opendde/data/template/template_finalizer.py`
- Modify: `opendde/data/template/template_featurizer.py`
- Modify: `runner/template_search.py`
- Create: `tests/test_explicit_templates.py`

**Interfaces:**
- Produces: `load_explicit_template_features(query_sequence, templates, *, base_dir, template_processor) -> list[Mapping[str, Any]]`.
- Produces: `finalize_template_hits(query_sequence, templates_path, template_featurizer, *, max_template_date) -> list[dict[str, Any]]`.
- Explicit list entries consume `mmcifPath`, `queryIndices`, and `templateIndices`; the referenced mmCIF must contain one protein chain.
- Automatic finalisation extracts the selected source chain to a single-chain mmCIF and emits those same three fields.

- [ ] **Step 1: Write failing explicit-template tests**

Use a tiny real single-chain mmCIF fixture and a monkeypatched
`TemplateHitProcessor._extract_template_features` to assert that the map passed
to the existing extractor is `{0: 5, 1: 6}`. Assert `templates: []` creates
empty template features without invoking the online hit featurizer. Assert an
explicit template bypasses the online hit featurizer and any date filtering.

For finalisation, provide a fake `TemplateHitFeaturizer.get_templates()` result
with a realigned hit and assert the serialised entry contains literal
`queryIndices`, `templateIndices`, and a single-chain mmCIF path.

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_explicit_templates.py -q`

Expected: failures show the explicit-template branch and finalizer are absent.

- [ ] **Step 3: Implement minimal explicit loading**

In `InferenceTemplateFeaturizer.make_template_feature`, prefer a non-`None`
`proteinChain.templates` value over legacy `templatesPath`. For every explicit
entry: read the mmCIF; reject `chainId`; require the parsed mmCIF to expose one
protein chain;
zip `queryIndices` and `templateIndices`; and call the existing processor's
`_extract_template_features()`. Add `template_sum_probs=[0.0]` because the model
assembly expects the same feature shape as searched templates. Do not apply a
release-date cutoff to explicit entries.

- [ ] **Step 4: Implement minimal hit finalisation**

Parse `.a3m` with `HmmsearchA3MParser` and `.hhr` with `HHRParser`; call the
existing `TemplateHitFeaturizer.get_templates()` exactly once. For every
selected realigned hit, serialise its non-gap mapping, retrieve the source
mmCIF through the configured hit processor, and extract the selected chain to a
single-chain mmCIF without renumbering the complete polymer sequence. Keep
selection order and maximum four behaviour from the existing featurizer. Extend `update_template_info()`
with optional finalisation arguments so the wrapper data stage can replace
`templatesPath` with `templates`; leave legacy callers unchanged.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_explicit_templates.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add opendde/data/template/template_finalizer.py opendde/data/template/template_featurizer.py runner/template_search.py tests/test_explicit_templates.py
git commit -m "feat: support explicit portable templates"
```

### Task 3: Separable data and prediction stages without hidden JSON

**Files:**
- Modify: `runner/fold_input.py`
- Modify: `runner/batch_inference.py`
- Modify: `runner/cli.py`
- Create: `tests/test_prediction_workflow.py`
- Modify: `tests/test_installation.py`

**Interfaces:**
- Produces: `prepare_input_jobs(input_path, out_dir, **pipeline_options) -> list[str]`.
- Produces: `run_prediction_workflow(..., run_data_pipeline: bool, run_inference: bool) -> list[str]`.
- `opendde pred` gains Click options `-D/--run_data_pipeline` and `-P/--run_inference`, both boolean and defaulting to `true`.
- `opendde pred` and `opendde prep` accept `--max_template_date`, defaulting to the existing OpenDDE cutoff `2021-09-30`; it affects automatic template search only.
- `opendde prep` calls `prepare_input_jobs()` and returns/prints prepared paths.

- [ ] **Step 1: Write failing workflow tests**

Assert data-only calls search/preparation but never `get_default_runner`; assert
inference-only passes the exact prepared JSON to `infer_predict` without
calling `preprocess_input`, `update_infer_json`, or `update_template_info`;
assert both-false fails before the output directory is created. Assert the only
JSON below the job directory is `<name>_data.json` and
`out/.opendde_preprocessed` does not exist.

Also assert that data-stage template finalisation receives the literal
`max_template_date` value while explicit `templates` entries do not invoke the
automatic template path.

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_prediction_workflow.py tests/test_installation.py -q`

Expected: failures show missing stage options/workflow and the old command path.

- [ ] **Step 3: Implement in-memory data preparation**

For each loaded job, resolve source-relative paths, deep-copy it, and call the
existing MSA/RNA/template functions on the in-memory job. Put their scratch
outputs under `tempfile.TemporaryDirectory()`. If `templates` is omitted or
`null`, run template search and finalise hits; if it is `[]` or a list, do not
search. Then pass the updated job to `write_prepared_job()`. Do not call the old
`preprocess_input()` and do not write a scratch JSON.

- [ ] **Step 4: Implement lazy inference**

Validate/discover the prepared JSONs before constructing the runner. Construct
one runner only when `run_inference` is true and at least one prepared input is
ready. Set `configs.input_json_path` to each prepared JSON and call the existing
`infer_predict()`. In inference-only mode, never call a data-stage function.
Close the runner in `finally`.

- [ ] **Step 5: Add CLI switches and prep delegation**

Add the two Click options to `predict` and pass them through. Keep all existing
model/data options. Add `--max_template_date` with default `2021-09-30` and pass
it only to the automatic data-stage template finaliser. Change `inputprep` to use the shared data-only function.
Keep the current lazy command registration names unchanged.

- [ ] **Step 6: Run tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_prediction_workflow.py tests/test_installation.py tests/test_inference_input_reliability.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add runner/fold_input.py runner/batch_inference.py runner/cli.py tests/test_prediction_workflow.py tests/test_installation.py tests/test_inference_input_reliability.py
git commit -m "feat: separate OpenDDE data and inference stages"
```

### Task 4: AF3-Pro-style prediction writer

**Files:**
- Modify: `runner/dumper.py`
- Modify: `runner/inference.py`
- Modify: `tests/test_dumper_reliability.py`

**Interfaces:**
- `DataDumper.dump()` continues receiving `pdb_id`, `seed`, prediction data,
  and atom metadata from the unchanged inference call site.
- Produces three job-level type directories and seed/sample filenames defined
  in the design spec.

- [ ] **Step 1: Replace output tests first**

Write a two-sample dumper test whose ranking scores are intentionally reversed.
Assert coordinates/sample zero still produce `seed-7_sample-0_model.cif` and
sample one produces `seed-7_sample-1_model.cif`. Assert summary and full-data
files use the matching original sample indices in their own directories and a
second seed does not overwrite the first.

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_dumper_reliability.py -q`

Expected: output-layout assertions fail against `seed_*/predictions/`.

- [ ] **Step 3: Implement direct type-directory writing**

Change `_get_dump_dir()` to return `<base>/<name>` and pass `seed` into each
filename. Write structure, summary, and optional full data directly to
`models/`, `summary_confidences/`, and `full_data/`. Iterate with
`enumerate(predictions)` and do not call `_get_ranker_indices()` for naming.
Keep B-factor annotation and existing serializers. Do not also write the old
tree.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_dumper_reliability.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add runner/dumper.py runner/inference.py tests/test_dumper_reliability.py
git commit -m "feat: write AF3-style prediction outputs"
```

### Task 5: User documentation, examples, and integrated regression

**Files:**
- Modify: `README.md`
- Modify: `docs/infer_json_format.md`
- Modify: `docs/msa_template_pipeline.md`
- Modify: `docs/inference_instructions.md`
- Create: `examples/example_wrapper_input.json`

**Interfaces:**
- Documents the exact CLI, JSON fields, prepared layout, unpaired-MSA
  replacement, template date semantics, and output layout implemented above.

- [ ] **Step 1: Update concise user documentation**

Add commands for data-only, inference-only, and both stages. Include one native
OpenDDE JSON with chain IDs, paired/unpaired paths, and an explicit template.
State that input filename does not control output; `name` does. State that
replacing only unpaired MSA is supported and paired rows remain merged into the
unpaired pool under the existing default.

- [ ] **Step 2: Add a small checked-in example**

Create a syntactically valid raw input named independently from its job name.
Use the literal protein sequence `MKTAYIAKQRQISFVKSHFSRQDILDLWQ` and no
machine-specific paths.

- [ ] **Step 3: Run documentation-adjacent and full CPU tests**

Run:

```bash
.venv/bin/python -m pytest tests -q -m "not network"
.venv/bin/ruff check runner/fold_input.py runner/batch_inference.py runner/dumper.py opendde/data/template tests/test_fold_input_bundle.py tests/test_explicit_templates.py tests/test_prediction_workflow.py tests/test_dumper_reliability.py
```

Expected: pytest and Ruff both exit zero.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/infer_json_format.md docs/msa_template_pipeline.md docs/inference_instructions.md examples/example_wrapper_input.json
git commit -m "docs: describe the OpenDDE wrapper workflow"
```
