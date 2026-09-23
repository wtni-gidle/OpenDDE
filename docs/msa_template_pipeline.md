# MSA, Template, and RNA MSA Pipeline


The `pred`/`prep` wrapper uses native OpenDDE job-list JSON and prepares one
portable bundle per job. Its model and MSA/template feature meanings are
unchanged.

## JSON fields

Protein:

```json
{
  "proteinChain": {
    "sequence": "MKTAYIAKQRQISFVKSHFSRQDILDLWQ",
    "count": 1,
    "id": ["A"],
    "pairedMsaPath": "assets/paired.a3m",
    "unpairedMsaPath": "assets/unpaired.a3m",
    "templates": [
      {
        "mmcifPath": "assets/template.cif",
        "queryIndices": [0, 1, 2],
        "templateIndices": [5, 6, 7]
      }
    ]
  }
}
```

RNA:

```json
{
  "rnaSequence": {
    "sequence": "GUAC",
    "count": 1,
    "id": ["R"],
    "unpairedMsaPath": "assets/rna_msa.a3m"
  }
}
```

Paths above are illustrative and resolve from the containing JSON's directory.
See the complete [wrapper example](../examples/example_wrapper_input.json);
its resource files are not included.

## Wrapper stages

For your own `input.json` with a job named `my_job`:

```bash
# Data only; choose feature searches explicitly
opendde pred -i input.json -o ./output -D true -P false \
  --use_msa true --use_template true --use_rna_msa true

# Inference only, consuming the prepared resources
opendde pred -i ./output/my_job/my_job_data.json -o ./output \
  -D false -P true --use_template true --use_rna_msa true

# Both stages (both switches default to true)
opendde pred -i input.json -o ./output \
  --use_template true --use_rna_msa true

# Data-only convenience with all three feature searches enabled
opendde prep -i input.json -o ./output
```

`pred` defaults to protein MSA enabled, templates and RNA MSA disabled. `prep`
enables all three where applicable and prints each prepared JSON path. Use
`pred -D true -P false` when selecting searches individually. Data-only does not
construct a model runner. Both stages disabled is an error.

The input filename does not control output identity; `name` does. A bundle with
protein chain ID `A` contains the supplied or generated files below:

```text
<out>/<name>/
├── <name>_data.json
└── msas/
    ├── <name>__A_pairedmsa.a3m
    ├── <name>__A_unpairedmsa.a3m
    └── <name>__A_template_0.cif
```

This shows the default `--compress_fold_input false`. With true the prepared suffixes are
`.a3m.zst` and `.cif.zst`; content magic, not the suffix, selects decompression.

When publication is enabled, the data stage writes one final JSON per job. Search scratch files live in
a temporary directory that is removed before return. Prepared paths are
relative to `_data.json`; move the whole job directory to move its MSA/template
resources. `FILE_` ligand files remain external absolute references; the caller
must keep them accessible or update their paths after moving to another machine.
An entity with `id: ["A", "B"]` shares one MSA pair named using `A`.

Inference-only accepts a prepared JSON directly, or a directory searched
recursively for `*_data.json` bundles. It does not search. By default it refreshes
the current supplied conditions; explicit `-J false` disables publication.
When every protein chain uses explicit templates or `templates: []`, template
use requires no HMMER, Kalign, template database, or PDBe access during inference,
including with `data.template.fetch_remote=false`. Legacy `templatesPath` is
rejected at the public boundary, even when `templates` is also present.
Model checkpoints and common
runtime assets are still required.

## Protein MSA and pairing

Protein MSA search uses the public ColabFold MMseqs2 API
(`https://api.colabfold.com`) when required MSA inputs are missing. The service is
shared and rate-limited; for batch runs, provide precomputed A3M files.

Set `MMSEQS_SERVICE_HOST_URL` to use a compatible self-hosted MMseqs2 endpoint.

MSA pairing uses species information from A3M headers. Supported examples:

```text
>UniRef100_<hit_name>_<species_or_taxonomy_id>/
>tr|ACCESSION|ID_SPECIES/START-END ...
>sp|ACCESSION|ID_SPECIES/START-END ...
```

OpenDDE extracts species identifiers from these supported UniRef/UniProt-style
descriptions; raw `OX=...` alone is not parsed. For complexes with distinct
protein sequences, cross-chain pairing uses those identifiers. Merely placing
an A3M in `pairedMsaPath` does not define an arbitrary row-by-row pairing
contract.

### Replace only unpaired MSA

After preparation, replace `<name>__A_unpairedmsa.a3m.zst` in place (plain A3M
text is accepted), leaving the paired A3M and template files unchanged.
Alternatively, edit `unpairedMsaPath`
in the prepared JSON; relative paths still resolve from that JSON. Run
inference-only (`-D false -P true`) with `--use_msa true` to consume the change.
The two MSA paths are read independently. The existing
`msa_pair_as_unpair=true` default also merges paired rows into the unpaired pool
and deduplicates them; replacing unpaired MSA does not disable this contribution
or change the paired input used for cross-chain pairing.

## Template search

Template search uses HMMER (`hmmbuild`, `hmmsearch`) against:

```text
$OPENDDE_ROOT_DIR/search_database/pdb_seqres_2022_09_28.fasta
```

With `--use_template true`, omitted or `null` `proteinChain.templates` requests
automatic selection during the data stage. `templates: []` disables templates
for that protein. A non-empty list uses the supplied mmCIFs and mappings without
automatic search. Each entry requires `mmcifPath`, `queryIndices`, and
`templateIndices`, with equal-length lists of zero-based residue indices. The
referenced mmCIF must contain exactly one protein chain. Automatic template hits
are extracted to this single-chain form after selection and realignment.

Run automatic preparation with explicit tools/database if needed:

```bash
opendde pred -i input.json -o ./output -D true -P false \
  --use_template true --max_template_date 2021-09-30 \
  --hmmsearch_binary_path /path/to/hmmsearch \
  --hmmbuild_binary_path /path/to/hmmbuild \
  --seqres_database_path /path/to/pdb_seqres_2022_09_28.fasta
```

Automatic preparation selects and realigns hits, obtains mmCIFs from local files or
PDBe, and writes explicit templates into the bundle. It respects
`--max_template_date` (default `2021-09-30`, also available on `prep`) and keeps
up to four selected templates. Explicit templates bypass the date cutoff,
matching AF3 semantics; model assembly still uses at most four templates.
Enable `--use_template true` again during inference to use prepared templates.

The EnsembleFold wrapper does not create or reuse parsed-template `.pkl` caches,
including temporary parsing caches. Preparation reads current CIF files and
inference rebuilds features from the current explicit templates. Custom template
featurizers passed to preparation or hit finalization must have
`template_cache_dir=None` or `''`; nonempty values raise a clear error.
The native `prot_template_cache_dir` configuration remains available to native
tools, but is not used by the prepared wrapper. This does not disable local CIF
files, release-date/obsolete-PDB metadata, or native cache tools, and does not
delete existing cache files. Search, filtering, alignment, and residue mappings
otherwise retain their existing behavior.

## RNA MSA

RNA MSA uses HMMER (`nhmmer`, `hmmalign`, `hmmbuild`) against NT-RNA, Rfam, and
RNAcentral databases.

```bash
opendde prep -i my_rna_job.json -o ./output \
  --nhmmer_binary_path /path/to/nhmmer \
  --hmmalign_binary_path /path/to/hmmalign \
  --hmmbuild_rna_binary_path /path/to/hmmbuild
```

Output:

```text
<out>/<name>/msas/<name>__R_unpairedmsa.a3m.zst
```

For RNA entity ID `R`, the prepared JSON points `unpairedMsaPath` to that file.
Inference requires `--use_rna_msa true` and `--use_msa true`.

## Legacy low-level `msa` and `mt` commands

The MSA diagnostic helper and portable template preparation command are:

```bash
opendde msa -i input.json -o ./output   # protein MSA only
opendde mt -i input.json -o ./output    # portable protein MSA + explicit templates
```

When JSON changes are needed, `msa` writes `*-update-msa.json` next to the
source JSON. `mt` now uses the same portable writer as `pred`/`prep`, returns one
`<out>/<name>/<name>_data.json` per job, and finalizes template hits before writing.
It no longer publishes legacy hit-file inputs. Low-level internal hit-processing
helpers remain available to the data pipeline, not as public inference inputs.

### Independent publication and temporary files

`pred -J/--write_input_json` controls public JSON/resource publication independently
of `-D` and `-P`. When omitted it defaults to true, including inference-only runs.
`-D true -J false` performs preparation in private scratch; resources stay alive
through inference and are then removed, including on failure. Returned paths in
this mode refer to the original inputs, not the deleted temporary snapshots.
`-D false -J true` republishes supplied conditions without searching. Both stages
disabled is an error. Distributed inference requires `-D false -J false`.

Publication always refreshes the snapshot, materializes inline or external
MSA/mmCIF as relative `msas/...` paths, stages resource reads before touching the
old bundle, and publishes JSON last. Caught publication failures restore replaced
files. This is not a crash-atomic multi-file transaction; concurrent writers to
the same bundle are unsupported. Old unreferenced files are not automatically
deleted. Private wrapper scratch prefers a writable `SLURM_TMPDIR`, then `TMPDIR`
or the system temporary directory; weights and CCD are outside its cleanup.

## Search databases

Databases are looked up under `$OPENDDE_ROOT_DIR/search_database/` and downloaded
when missing. The default download source is the AlphaFold v3.0 `.fasta.zst`
archives, which are decompressed to the `.fasta` runtime files below.

| Database | Used by | Size |
| --- | --- | --- |
| `pdb_seqres_2022_09_28.fasta` | template | ~220 MB |
| `rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta` | RNA MSA | ~220 MB |
| `nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta` | RNA MSA | ~75 GB |
| `rnacentral_active_seq_id_90_cov_80_linclust.fasta` | RNA MSA | ~13 GB |

Override downloads:

- `OPENDDE_SEARCH_DATABASE_URL`: root for all search databases.
- `OPENDDE_PDB_SEQRES_URL`, `OPENDDE_RFAM_DB_URL`, `OPENDDE_NT_RNA_DB_URL`,
  `OPENDDE_RNACENTRAL_DB_URL`: individual database URLs.

## Dependencies

- Protein MSA: public ColabFold MMseqs2 API or `MMSEQS_SERVICE_HOST_URL`.
- Automatic template preparation: `hmmbuild`, `hmmsearch`, and `kalign` for
  realignment when needed.
- RNA MSA: `nhmmer`, `hmmalign`, `hmmbuild`.
- Search database decompression: `zstd` command, or the optional Python
  `zstandard` package for Python auto-downloads.
- Legacy hit-file inference: `kalign` when realignment is needed; explicit
  prepared templates do not use it.
