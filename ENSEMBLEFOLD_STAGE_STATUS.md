# EnsembleFold wrapper stage status — 2026-09-24

Defaults: `write_input_json=true`, `compress_fold_input=false`, and
`compress_full_confidence=false`. Full data remains enabled by default.
The sample publisher stages files and rolls back caught publication errors;
this is not a whole-seed, concurrent-writer, or power-loss transaction.
See `docs/inference_instructions.md` and `docs/msa_template_pipeline.md`.

Earlier-stage CPU/offline regression: 853 tests and 70 subtests, excluding
network and smoke tests; no GPU/model equivalence claim. These counts predate
the diagnostics follow-up below.

The proposed mandatory template declaration was cancelled: existing omission/
null and template-use switches remain unchanged. Legacy MSA directory inputs still
exist, and external `FILE_` ligands are not included in portable MSA/template
bundling. Distributed data/publication restrictions remain intentional.
Skip means required files exist and are non-empty, not readable/valid content.

2026-09-23 diagnostics follow-up: automatic hit finalization now reports all
native errors and warnings with task/chain/source context and lists the retained
template identities/count, including zero. Normal empty/filtered results are
distinguished from results containing errors. The existing continuation policy
is preserved: all returned failures can still yield `templates: []`; this change
does not make that case abort preparation. Native selection and feature
algorithms are unchanged.

On stat CPU, job 92096 confirmed all five new diagnostic cases failed before
implementation; job 92098 passed those five plus 31 selected regression cases
(36 passed). Final broader post-change regression (job 92162): 858 passed,
70 subtests, 4 skipped and 4 deselected. One old test double needed its signature
updated for diagnostic context; the existing cutoff/scratch assertions remain.
No network search or GPU/model inference was run for this follow-up.

Final verification (2026-09-24): 858 CPU tests and 70 subtests passed again
(job 93946; 4 skipped, 4 deselected). Real-model GPU inference was attempted
in allocation 91413 and correctly exited nonzero: the installed PyTorch
2.7.1+cu126 does not support that Blackwell sm_120 GPU (`no kernel image`).
GPU acceptance remains blocked by the environment, not claimed passed.
No new blocking issue or material redundancy was found in the changed wrapper
paths; native code and installed environments were not modified.
