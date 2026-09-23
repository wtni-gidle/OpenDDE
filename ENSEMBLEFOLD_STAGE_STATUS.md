# EnsembleFold wrapper stage status — 2026-09-23

Defaults: `write_input_json=true`, `compress_fold_input=false`, and
`compress_full_confidence=false`. Full data remains enabled by default.
The sample publisher stages files and rolls back caught publication errors;
this is not a whole-seed, concurrent-writer, or power-loss transaction.
See `docs/inference_instructions.md` and `docs/msa_template_pipeline.md`.

CPU/offline regression: 853 tests and 70 subtests, excluding network and smoke
tests; no GPU/model equivalence claim.

The proposed mandatory template declaration was cancelled: existing omission/
null and template-use switches remain unchanged. Automatic-template technical
errors can still be lost when hits are finalized to an empty list; this remains
an unresolved diagnostic/condition risk. Legacy MSA directory inputs still
exist, and external `FILE_` ligands are not included in portable MSA/template
bundling. Distributed data/publication restrictions remain intentional.
Skip means required files exist and are non-empty, not readable/valid content.
