#!/usr/bin/env bash
# Run a pipeline over whole datasets, several samples at a time.
#
#   zeroshot/run_parallel.sh --pipeline default --model MODEL \
#       --datasets test_eccv,test_itercad --out OUT [--jobs 8] [--samples IDS] \
#       [--dry-run] [OVERRIDES...]
#
# --pipeline is default (the staged pipeline), native or native2; --datasets takes
# input configs separated by commas; --samples keeps only the sample ids it lists,
# separated by commas; OVERRIDES are Hydra overrides for every sample. Each
# sample is one single-run process landing in OUT/<dataset>/<sample_id>, with
# its console output in OUT/_logs/<dataset>/<sample_id>.log. Completed samples
# are skipped (on_existing=retry), so the same command resumes an interrupted
# sweep.
#
# Start it in a terminal (a zellij pane, say) or with setsid, so the sweep has a
# process group of its own: Ctrl-C, or killing that group (OUT/_logs/pgid), stops
# every sample, and SIGUSR1 or SIGUSR2 to its xargs runs one sample more or fewer.
set -euo pipefail

fail() { echo "$*" >&2; exit 1; }
needs_value() { (($# >= 2)) || fail "$1 needs a value"; }
jobs=8
while (($#)); do
  case $1 in
    --pipeline) needs_value "$@"; pipeline=$2; shift 2 ;;
    --model) needs_value "$@"; model=$2; shift 2 ;;
    --datasets | --dataset) needs_value "$@"; datasets=$2; shift 2 ;;
    --out) needs_value "$@"; out=$2; shift 2 ;;
    --jobs) needs_value "$@"; jobs=$2; shift 2 ;;
    --samples) needs_value "$@"; only=$2; shift 2 ;;
    --dry-run) export DRY_RUN=1; shift ;;
    -h | --help) sed -n '2,/^set /{/^set /d; s/^# \{0,1\}//p}' "$0"; exit 0 ;;
    --) shift; break ;;
    -*) fail "unknown option $1 (see --help)" ;;
    *) break ;;
  esac
done
for option in pipeline model datasets out; do
  [[ -n ${!option:-} ]] || fail "--$option is required (see --help)"
done
out=$(realpath -m "$out")
cd "$(dirname "$0")/.."

case $pipeline in
  default) module=zeroshot.run_pipeline ;;
  native | native2) module=zeroshot.pipeline_$pipeline ;;
  *) fail "unknown pipeline $pipeline" ;;
esac
[[ -f zeroshot/configs/model/$model.yaml ]] || fail "no model config $model"
for dataset in ${datasets//,/ }; do
  [[ -f zeroshot/configs/input/$dataset.yaml ]] || fail "no input config $dataset"
done
# xargs -P 0 would start every sample at once.
[[ $jobs =~ ^[1-9][0-9]*$ ]] || fail "--jobs must be a positive integer"

# The sandbox mounts ${CONDA_PREFIX} as its Python, so it must be this one.
python -c "import cadquery" 2>/dev/null || fail "activate the drawing2cad environment"
CONDA_PREFIX=$(python -c "import sys; print(sys.prefix)")
export CONDA_PREFIX
# Software rendering needs no GPU memory and draws the same projections.
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow
export module model out

# Writes "<dataset> <sample_id>" for each file the dataset's target path matches.
list_samples() {
  local pattern name target id
  pattern=$(sed -n 's/^ *target_step_path: *//p' "zeroshot/configs/input/$1.yaml")
  name=${pattern##*/}
  for target in ${pattern//'${sample.sample_id}'/*}; do
    [[ -e $target ]] || continue
    id=${target##*/}
    id=${id#"${name%%'${sample.sample_id}'*}"}
    echo "$1 ${id%"${name#*'${sample.sample_id}'}"}"
  done
}

# OVERRIDES..., dataset, sample_id
run_one() {
  local dataset=${*: -2:1} id=${*: -1} status=0
  local log="$out/_logs/$dataset/$id.log"
  local command=(python -m "$module" "model=$model" "input=$dataset"
    "artifact_root=$out/$dataset" "sample.sample_id='$id'" on_existing=retry
    "${@:1:$#-2}")
  if [[ -n ${DRY_RUN:-} ]]; then
    echo "${command[*]}"
    return
  fi
  mkdir -p "${log%/*}"
  echo "$(date '+%F %T') start $dataset/$id"
  printf '\n===== %s start\n' "$(date '+%F %T')" >>"$log"
  "${command[@]}" >>"$log" 2>&1 || status=$?
  printf '\n===== %s exit %s\n' "$(date '+%F %T')" "$status" >>"$log"
  echo "$(date '+%F %T') exit $status $dataset/$id"
}
export -f run_one

# Datasets take turns, so early results cover all of them.
samples=$(for dataset in ${datasets//,/ }; do list_samples "$dataset"; done |
  awk '{ print n[$1]++ "\t" NR "\t" $0 }' | sort -n -k1,1 -k2,2 | cut -f3-)
if [[ -n ${only:-} ]]; then
  samples=$(grep -wFf <(tr , '\n' <<<"$only") <<<"$samples") ||
    fail "--samples matches no sample of $datasets"
fi

if [[ -z ${DRY_RUN:-} ]]; then
  # A pipeline run carries on through SIGINT, so Ctrl-C ends the whole group.
  trap 'trap - TERM; kill -TERM 0' INT
  mkdir -p "$out/_logs"
  ps -o pgid= -p $$ | tr -d ' ' >"$out/_logs/pgid"
  echo "$(wc -l <<<"$samples") samples, $jobs at a time; logs in $out/_logs"
  # Available memory each minute, to judge how many samples fit.
  while sleep 60; do
    echo "$(date '+%F %T') $(awk '/MemAvailable/ {printf "%.1f", $2 / 2^20}' /proc/meminfo) GiB"
  done >>"$out/_logs/memory.log" &
  trap 'kill $! 2>/dev/null' EXIT
fi

xargs -P "$jobs" -L 1 bash -c 'run_one "$@"' _ "$@" <<<"$samples"
