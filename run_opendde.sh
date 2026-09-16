#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_opendde.sh -i INPUT -o OUTPUT [options] [-- EXTRA_OPENDDE_ARGS...]

Required:
  -i PATH   Input JSON file or directory.
  -o PATH   Output directory.

Wrapper options:
  -D BOOL   Run data pipeline (default: true).
  -P BOOL   Run inference (default: true).
  -r LIST   One seed or comma-separated seeds.
  -s INT    Diffusion samples per seed (default: 5).
  -m DATE   Automatic-template cutoff (default: 2021-09-30).
  -S BOOL   Skip complete job/seed outputs (default: false).
  -w BOOL   write_now compatibility flag (default: true).
  -z BOOL   --compress_fold_input (default: true).
  -f BOOL   --compress_full_confidence (default: false).
  -a BOOL   --need_atom_confidence (default: true).
  -h        Show this help.

Set OPENDDE_BIN to the OpenDDE command from your configured environment;
it defaults to "opendde". Arguments after -- are forwarded unchanged.
EOF
}

input_path=""
output_dir=""
run_data_pipeline="true"
run_inference="true"
seeds=""
sample="5"
max_template_date="2021-09-30"
skip="false"
write_now="true"
compress_fold_input="true"
compress_full_confidence="false"
need_atom_confidence="true"

while getopts ":i:o:D:P:r:s:m:S:w:z:f:a:h" option; do
    case "$option" in
        i) input_path=$OPTARG ;;
        o) output_dir=$OPTARG ;;
        D) run_data_pipeline=$OPTARG ;;
        P) run_inference=$OPTARG ;;
        r) seeds=$OPTARG ;;
        s) sample=$OPTARG ;;
        m) max_template_date=$OPTARG ;;
        S) skip=$OPTARG ;;
        w) write_now=$OPTARG ;;
        z) compress_fold_input=$OPTARG ;;
        f) compress_full_confidence=$OPTARG ;;
        a) need_atom_confidence=$OPTARG ;;
        h) usage; exit 0 ;;
        :) echo "Error: -$OPTARG requires a value." >&2; usage >&2; exit 2 ;;
        \?) echo "Error: unknown option -$OPTARG." >&2; usage >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
if [[ ${1:-} == "--" ]]; then
    shift
fi

if [[ -z $input_path || -z $output_dir ]]; then
    echo "Error: -i and -o are required." >&2
    usage >&2
    exit 2
fi
if [[ ! -e $input_path ]]; then
    echo "Error: input path does not exist: $input_path" >&2
    exit 2
fi

opendde_bin=${OPENDDE_BIN:-opendde}
if ! command -v "$opendde_bin" >/dev/null 2>&1; then
    echo "Error: OpenDDE executable not found: $opendde_bin" >&2
    exit 127
fi

command_args=(
    pred
    --input "$input_path"
    --out_dir "$output_dir"
    --run_data_pipeline "$run_data_pipeline"
    --run_inference "$run_inference"
    --sample "$sample"
    --max_template_date "$max_template_date"
    --skip "$skip"
    --write_now "$write_now"
    --compress_fold_input "$compress_fold_input"
    --compress_full_confidence "$compress_full_confidence"
    --need_atom_confidence "$need_atom_confidence"
)
if [[ -n $seeds ]]; then
    command_args+=(--seeds "$seeds")
fi
command_args+=("$@")

printf 'Running:'
printf ' %q' "$opendde_bin" "${command_args[@]}"
printf '\n'
exec "$opendde_bin" "${command_args[@]}"
