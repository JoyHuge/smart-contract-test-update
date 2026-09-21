#!/usr/bin/env bash
# Run one record, method, and model combination in a prepared project image.
set -Eeuo pipefail

: "${SCT_RECORD_INDEX:?SCT_RECORD_INDEX is required}"
: "${SCT_PROJECT_DIR:?SCT_PROJECT_DIR is required}"
: "${SCT_DATASET_PATH:?SCT_DATASET_PATH is required}"

SCT_AGENT_DIR="${SCT_AGENT_DIR:-/opt/sct/agent}"
SCT_RESULT_DIR="${SCT_RESULT_DIR:-/results}"
SCT_JOB_KIND="${SCT_JOB_KIND:-experiment}"

if [[ ! -d "$SCT_PROJECT_DIR" ]]; then
  echo "Project directory does not exist in the image: $SCT_PROJECT_DIR" >&2
  exit 66
fi
if [[ ! -f "$SCT_DATASET_PATH" ]]; then
  echo "Dataset mount is missing: $SCT_DATASET_PATH" >&2
  exit 66
fi

# Images may provide an optional project-specific setup hook.
if [[ -x /opt/sct/job-setup.sh ]]; then
  /opt/sct/job-setup.sh
fi

cleanup() {
  if [[ -x /opt/sct/job-cleanup.sh ]]; then
    /opt/sct/job-cleanup.sh || true
  fi
}
trap cleanup EXIT

export SCT_SOURCES_CACHE_DIR="${SCT_RESULT_DIR}/.cache/sources"
export SCT_GROUND_TRUTH_TEST_CACHE_PATH="${SCT_RESULT_DIR}/.cache/ground-truth-tests.json"
export SCT_PARALLEL_MODELS=0
export PYTHONPATH="${SCT_AGENT_DIR}:${PYTHONPATH:-}"

mkdir -p "$SCT_RESULT_DIR"
cd "$SCT_RESULT_DIR"

if [[ "$SCT_JOB_KIND" == "prepare" ]]; then
  python3 -u "${SCT_AGENT_DIR}/runtime/prepare_sample.py"
  exit $?
fi

if [[ "$SCT_JOB_KIND" != "experiment" ]]; then
  echo "Unknown job kind: $SCT_JOB_KIND" >&2
  exit 64
fi

: "${SCT_METHOD:?SCT_METHOD is required for experiment jobs}"
: "${SCT_MODEL:?SCT_MODEL is required for experiment jobs}"
export SCT_MODELS="$SCT_MODEL"

case "$SCT_METHOD" in
  SCT-Agent)
    export SCT_SCT_MODES=full
    python3 -u "${SCT_AGENT_DIR}/main_SCT.py"
    ;;
  SCT-Agent-wo-TestRunner)
    export SCT_SCT_MODES=wo_test_runner
    python3 -u "${SCT_AGENT_DIR}/main_SCT.py"
    ;;
  SCT-Agent-wo-AST-wo-TestRunner)
    export SCT_SCT_MODES=wo_ast_wo_test_runner
    python3 -u "${SCT_AGENT_DIR}/main_SCT.py"
    ;;
  SDG)
    python3 -u "${SCT_AGENT_DIR}/main_SDG.py"
    ;;
  SSR)
    python3 -u "${SCT_AGENT_DIR}/main_SSR.py"
    ;;
  *)
    echo "Unknown method: $SCT_METHOD" >&2
    exit 64
    ;;
esac

eval_path=$(find . -type f -name eval.json -path '*/output_*/*' -print -quit)
if [[ -z "$eval_path" ]]; then
  echo "The experiment did not produce eval.json." >&2
  exit 1
fi
