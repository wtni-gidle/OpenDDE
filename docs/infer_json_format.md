# Inference JSON Format


OpenDDE input is a JSON file whose top-level value is a non-empty list of jobs.
It uses AlphaFold Server-style entity keys (`proteinChain`, `dnaSequence`,
`rnaSequence`, `ligand`, `ion`), not the single-job `alphafold3` dialect.

Minimal job:

```json
[
  {
    "name": "example_job",
    "modelSeeds": [101],
    "sequences": [
      {
        "proteinChain": {
          "sequence": "ACDEFGHIKLMNPQRSTVWY",
          "count": 1
        }
      }
    ]
  }
]
```

`covalent_bonds` is optional and is omitted here; see the section below for when
to add it.

Job fields:

| Field | Required | Meaning |
| --- | :---: | --- |
| `name` | Yes | Job name used in output paths. |
| `sequences` | Yes | List of entities. Each item has exactly one entity key. |
| `modelSeeds` | No | Default seeds for the job. Overridden by `--seeds`; if neither is set, a random seed is sampled. |
| `covalent_bonds` | No | Explicit covalent links between entities. |

Every entity has `count`. Optional `id` is a list of chain IDs; its length must
match `count`.

Each job's `name` determines `<out>/<name>/` and `<name>_data.json`; the input
JSON filename has no naming semantics. Names must be safe path components and
unique across the input collection. Protein MSA/template paths and RNA MSA paths
resolve relative to the JSON file that contains them, not the launch directory.
Prepared bundles use the same rule and can be moved as complete job directories.

## `proteinChain`

```json
{
  "proteinChain": {
    "sequence": "ACDEFGHIKLMNPQRSTVWY",
    "count": 1,
    "id": ["A"],
    "modifications": [
      {"ptmType": "CCD_MSE", "ptmPosition": 1}
    ],
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

- `sequence`: 20 standard amino-acid letters plus `X`.
- `ptmType`: CCD code prefixed with `CCD_`; `ptmPosition` is 1-based.
- `pairedMsaPath`, `unpairedMsaPath`: optional protein A3M files.
- An entity with `count: 2` and `id: ["A", "B"]` shares one paired/unpaired
  MSA pair, named using its first ID (`A`) in the prepared bundle.

### Protein templates

With `--use_template true`, `templates` has three states:

| Value | Behavior |
| --- | --- |
| Omitted or `null` | Automatic template search/selection during the data stage. |
| `[]` | Use no templates for this protein. |
| Non-empty list | Use the supplied explicit templates without automatic search. |

Each explicit entry requires `mmcifPath`, `queryIndices`, and `templateIndices`.
The two index lists have equal length and contain zero-based residue indices in
the query and template chain sequences, respectively. Each mmCIF must contain
exactly one protein chain, as in AF3 input. Automatically selected templates are
extracted from their source PDB entries and written as single-chain mmCIF files.

Automatic selection respects `--max_template_date YYYY-MM-DD` (default
`2021-09-30`). Explicit templates bypass the release-date cutoff, matching AF3
semantics; the existing model assembly still uses at most four templates.
Set `--use_template true` during inference to consume them.

Legacy `templatesPath` is rejected, including empty/null values and combinations
with `templates`. Supply `templates` in the main JSON. Each entry accepts either
inline `mmcif` text or `mmcifPath`, never both, and a single protein chain (no
`chainId`). `queryIndices` and `templateIndices` must be equally long lists of
unique, nonnegative integers within their full polymer sequences; unresolved
residues retain their positions. Automatic search hits remain private data-stage
intermediates, finalized into this same explicit format before prediction.

For each MSA channel, inline `pairedMsa`/`unpairedMsa` and the corresponding
`...Path` are alternatives. An empty inline string explicitly disables that
channel. A supplied missing file is an error, not permission to search again.
A protein with either channel supplied is protected from automatic MSA search
writeback, even when another protein needs searching. Native search context and
pairing rules are unchanged. RNA has only the native unpaired route.

The complete [wrapper example](../examples/example_wrapper_input.json) has job
name `wrapper_demo`, independent of its filename. Its relative resource files
are illustrative and are not included; provide matching A3M/mmCIF files before
using it.

## `dnaSequence`

```json
{
  "dnaSequence": {
    "sequence": "GATTACA",
    "count": 1,
    "id": ["D"],
    "modifications": [
      {"modificationType": "CCD_6MA", "basePosition": 2}
    ]
  }
}
```

- Supported documented letters: `A`, `T`, `G`, `C`, `N`, `X`.
- DNA is single-stranded; add another `dnaSequence` for the other strand.
- `basePosition` is 1-based.

## `rnaSequence`

```json
{
  "rnaSequence": {
    "sequence": "GUAC",
    "count": 1,
    "id": ["R"],
    "modifications": [
      {"modificationType": "CCD_5MC", "basePosition": 4}
    ],
    "unpairedMsaPath": "assets/rna_msa.a3m"
  }
}
```

- Supported documented letters: `A`, `U`, `G`, `C`, `N`, `X`.
- `unpairedMsaPath` is optional and used only with `--use_rna_msa true`.

## `ligand`

```json
{
  "ligand": {
    "ligand": "CCD_ATP",
    "count": 1,
    "id": ["L"]
  }
}
```

`ligand` can be:

- A CCD code prefixed with `CCD_`, e.g. `CCD_ATP`.
- Multiple CCD codes joined by underscores, e.g. `CCD_NAG_BMA_BGC`.
- A 3D ligand file prefixed with `FILE_` (`.pdb`, `.sdf`, `.mol`, `.mol2`).
- A SMILES string.

For `FILE_` ligands, the path after the prefix resolves relative to the owning
JSON file. Preparation keeps an absolute `FILE_` reference in `_data.json`;
the ligand file stays external and is not copied into `msas/`. Keep that file
accessible, or update the reference when moving to another machine.

## `ion`

```json
{
  "ion": {
    "ion": "MG",
    "count": 2,
    "id": ["M", "N"]
  }
}
```

Ion codes are CCD component names without the `CCD_` prefix.

## `covalent_bonds`

```json
"covalent_bonds": [
  {
    "entity1": "1",
    "copy1": 1,
    "position1": "2",
    "atom1": "SG",
    "entity2": "2",
    "copy2": 1,
    "position2": "1",
    "atom2": "C1"
  }
]
```

Fields:

- `entity1`, `entity2`: 1-based indices in `sequences`.
- `copy1`, `copy2`: optional 1-based copy indices.
- `position1`, `position2`: 1-based residue/ligand-part positions.
- `atom1`, `atom2`: atom names. Integer references are also accepted for mapped
  SMILES or file ligands.

Use `entity1`/`entity2` for new inputs. The old `left_entity`/`right_entity`
style is accepted for compatibility.

## Unsupported `constraint`

The inference-only build ignores legacy `constraint` fields. Use
`covalent_bonds` for supported covalent links.

## Output layout

The data stage (`opendde pred -D true -P false`, or `opendde prep`) writes a
single-job JSON and copies supplied/generated resources:

```text
<out>/<name>/
├── <name>_data.json
└── msas/
    ├── <name>__A_pairedmsa.a3m
    ├── <name>__A_unpairedmsa.a3m
    └── <name>__A_template_0.cif
```

By default the prepared resource names add `.zst` (`.a3m.zst` and `.cif.zst`).
Use `--compress_fold_input false` for the plain layout shown above. Compression
is detected from file content, so a plain replacement MSA remains readable even
when its prepared path still ends in `.zst`.

Resources appear only when supplied or generated. Unknown JSON fields and
unrelated entity fields are preserved. RNA MSA uses the same `msas/` directory
and `<name>__<entity-ID>_unpairedmsa.a3m` naming.

You may replace just the prepared unpaired A3M file, or its `unpairedMsaPath`,
then run `pred -D false -P true` on the prepared JSON. The paired MSA and
templates remain independent inputs. The default `msa_pair_as_unpair=true`
also contributes paired rows to the unpaired pool with deduplication. See
[MSA pairing details](msa_template_pipeline.md#protein-msa-and-pairing).

Prediction writes directly under the requested output directory:

```text
<out>/<name>/
├── models/seed-101_sample-0_model.cif
├── summary_confidences/seed-101_sample-0_summary_confidences.json
└── full_data/seed-101_sample-0_full_data.json
```

Sample numbers are original diffusion sample indices, not confidence ranks.
Seeds are included in filenames. `full_data/` is written only with
`--need_atom_confidence true` (the default). If inference uses a different output
directory from preparation, the current prepared bundle is published there by
default; use `--write_input_json false` to suppress that snapshot.

Detailed confidence defaults to JSON. Set
`--compress_full_confidence true` to select compressed NPZ; model CIFs and
summary-confidence JSONs retain their existing formats.

The summary JSON includes confidence metrics such as `plddt`, `gpde`, `ptm`,
`iptm`, clash flags, and `ranking_score` when available.
