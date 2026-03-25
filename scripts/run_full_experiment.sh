#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTAINER_REPO_DIR="${CONTAINER_REPO_DIR:-/workspace/ATES}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${REPO_DIR}/config/ates/default.yaml}"
CONTAINER_EXPERIMENT_CONFIG="${CONTAINER_EXPERIMENT_CONFIG:-${CONTAINER_REPO_DIR}/config/ates/default.yaml}"
DATA_ROOT="${DATA_ROOT:-/mnt/data/FishEye8K}"
COMPOSE_FILE="${COMPOSE_FILE:-${REPO_DIR}/compose.yaml}"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-ates}"

BUILD_CONTAINERS="${BUILD_CONTAINERS:-1}"
START_CONTAINERS="${START_CONTAINERS:-1}"
RUN_AUTOMATIC_V1="${RUN_AUTOMATIC_V1:-1}"
DRY_RUN="${DRY_RUN:-0}"

BASE_SERVICE="${BASE_SERVICE:-base}"
PSEUDO_SERVICE="${PSEUDO_SERVICE:-pseudo}"
GEN_SERVICE="${GEN_SERVICE:-gen}"
DPO_SERVICE="${DPO_SERVICE:-dpo}"

SIMPLETUNER_COMMAND="${SIMPLETUNER_COMMAND:-ENV=simpletuner bash scripts/simpletuner_train.sh}"
FLUX_ADAPTER_GLOB="${FLUX_ADAPTER_GLOB:-${REPO_DIR}/ckpt/flux/**/checkpoint-*}"
AUTOMATIC_REPHRASER_GLOB="${AUTOMATIC_REPHRASER_GLOB:-${REPO_DIR}/ckpt/llama/llama_dpo_fisheye8k_with_naive_v0/checkpoint-*}"


log() {
    printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}


require_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Missing required command: $1" >&2
        exit 1
    }
}


ensure_python_package_set() {
    local service="$1"
    local import_check="$2"
    local install_command="$3"

    if [[ "${DRY_RUN}" == "1" ]]; then
        log "Would verify Python deps in service ${service}: ${import_check}"
        return 0
    fi

    if docker compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT_NAME}" exec -T "${service}" bash -lc "cd '${CONTAINER_REPO_DIR}' && python -c \"${import_check}\"" >/dev/null 2>&1; then
        return 0
    fi

    docker_exec_repo "${service}" "${install_command}"
}


run() {
    log "$*"
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    "$@"
}


docker_exec_repo() {
    local service="$1"
    shift
    local command="$*"
    run docker compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT_NAME}" exec -T "${service}" bash -lc "cd '${CONTAINER_REPO_DIR}' && ${command}"
}


compose_service_exists() {
    docker compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT_NAME}" config --services | grep -Fxq "$1"
}


compose_ps_running() {
    docker compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT_NAME}" ps --status running --services | grep -Fxq "$1"
}


maybe_build_and_start_containers() {
    if [[ "${BUILD_CONTAINERS}" == "1" ]]; then
        run docker compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT_NAME}" build
    fi

    if [[ "${START_CONTAINERS}" == "1" ]]; then
        run docker compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT_NAME}" up -d \
            "${BASE_SERVICE}" "${PSEUDO_SERVICE}" "${GEN_SERVICE}" "${DPO_SERVICE}"
    fi

    for service in "${BASE_SERVICE}" "${PSEUDO_SERVICE}" "${GEN_SERVICE}" "${DPO_SERVICE}"; do
        compose_service_exists "${service}" || {
            echo "Service not found in compose file: ${service}" >&2
            exit 1
        }
        compose_ps_running "${service}" || {
            echo "Service is not running: ${service}" >&2
            exit 1
        }
    done
}


update_config_value() {
    local dotted_key="$1"
    local raw_value="$2"
    if [[ "${DRY_RUN}" == "1" ]]; then
        log "Would set ${dotted_key}=${raw_value} in ${EXPERIMENT_CONFIG}"
        return 0
    fi

    python3 - "${EXPERIMENT_CONFIG}" "${dotted_key}" "${raw_value}" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
dotted_key = sys.argv[2]
raw_value = sys.argv[3]

value = None if raw_value == "__NULL__" else raw_value

with config_path.open("r", encoding="utf-8") as handle:
    data = yaml.safe_load(handle)

target = data
parts = dotted_key.split(".")
for key in parts[:-1]:
    target = target[key]
target[parts[-1]] = value

with config_path.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(data, handle, sort_keys=False)
PY
}


find_best_checkpoint() {
    local pattern="$1"
    python3 - "${pattern}" <<'PY'
import glob
import os
import re
import sys

pattern = sys.argv[1]
candidates = glob.glob(pattern, recursive=True)
if not candidates:
    sys.exit(1)

def sort_key(path: str):
    base = os.path.basename(path)
    match = re.search(r'(\d+)(?!.*\d)', base)
    numeric = int(match.group(1)) if match else -1
    return (numeric, path)

print(sorted(candidates, key=sort_key)[-1])
PY
}


find_latest_checkpoint_dir() {
    local pattern="$1"
    python3 - "${pattern}" <<'PY'
import glob
import os
import re
import sys

pattern = sys.argv[1]
candidates = [path for path in glob.glob(pattern, recursive=True) if os.path.isdir(path)]
if not candidates:
    sys.exit(1)

def sort_key(path: str):
    base = os.path.basename(path)
    match = re.search(r'checkpoint-(\d+)', base)
    numeric = int(match.group(1)) if match else -1
    return (numeric, path)

print(sorted(candidates, key=sort_key)[-1])
PY
}


require_file() {
    local path="$1"
    [[ -e "${path}" ]] || {
        echo "Required path not found: ${path}" >&2
        exit 1
    }
}


main() {
    require_cmd docker
    require_cmd python3
    require_file "${EXPERIMENT_CONFIG}"
    require_file "${COMPOSE_FILE}"
    require_file "${REPO_DIR}/tools/run_experiment.py"

    maybe_build_and_start_containers

    log "Bootstrapping Python environments in compose services"
    ensure_python_package_set "${BASE_SERVICE}" "import yaml, ultralytics, transformers, accelerate, datasets, peft, timm" "pip install -e ."
    ensure_python_package_set "${PSEUDO_SERVICE}" "import yaml, mmengine, mmdet, pycocotools, ultralytics, timm, matplotlib" "pip install -e ."
    ensure_python_package_set "${GEN_SERVICE}" "import yaml, accelerate, diffusers, transformers, lycoris" "pip install -e ."
    ensure_python_package_set "${DPO_SERVICE}" "import yaml, accelerate, datasets, transformers, peft, trl" "pip install -e ."

    log "Updating experiment config with DATA_ROOT=${DATA_ROOT}"
    update_config_value "paths.data_root" "${DATA_ROOT}"

    log "Rendering derived configs"
    docker_exec_repo "${BASE_SERVICE}" "python tools/run_experiment.py --experiment-config '${CONTAINER_EXPERIMENT_CONFIG}' render-configs"

    log "Ensuring Co-DETR pretrained checkpoint is present"
    docker_exec_repo "${PSEUDO_SERVICE}" "mkdir -p ckpt/codetr && test -f ckpt/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth || wget -P ckpt/codetr/ https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth"

    log "Stage 1: Train Co-DETR pseudo-labeler in ${PSEUDO_SERVICE}"
    docker_exec_repo "${PSEUDO_SERVICE}" "python tools/run_experiment.py --experiment-config '${CONTAINER_EXPERIMENT_CONFIG}' mmdet-train"

    log "Updating config with best Co-DETR checkpoint"
    local codetr_ckpt
    codetr_ckpt="$(find_best_checkpoint "${REPO_DIR}/ckpt/codetr/codetr_fisheye8k/best_coco_bbox_mAP_epoch_*.pth")"
    update_config_value "checkpoints.codetr_finetuned" "${codetr_ckpt}"

    log "Stage 1b: Temporary pseudo labels in ${PSEUDO_SERVICE}"
    docker_exec_repo "${PSEUDO_SERVICE}" "python tools/run_experiment.py --experiment-config '${CONTAINER_EXPERIMENT_CONFIG}' obtain-tmp-pseudo"

    log "Stage 1c: Estimate optimal thresholds in ${PSEUDO_SERVICE}"
    docker_exec_repo "${PSEUDO_SERVICE}" "python tools/run_experiment.py --experiment-config '${CONTAINER_EXPERIMENT_CONFIG}' estimate-threshold"

    log "Stage 2: Train Flux / SimpleTuner in ${GEN_SERVICE}"
    docker_exec_repo "${GEN_SERVICE}" "${SIMPLETUNER_COMMAND}"

    log "Updating config with latest Flux adapter checkpoint"
    local flux_ckpt
    flux_ckpt="$(find_latest_checkpoint_dir "${FLUX_ADAPTER_GLOB}")"
    update_config_value "checkpoints.flux_adapter" "${flux_ckpt}"

    log "Stage 3: Extract captions and diverse rephrases in ${BASE_SERVICE}"
    docker_exec_repo "${BASE_SERVICE}" "python tools/run_experiment.py --experiment-config '${CONTAINER_EXPERIMENT_CONFIG}' extract-and-rephrase"

    log "Stage 4: Synthesize images in ${GEN_SERVICE}"
    docker_exec_repo "${GEN_SERVICE}" "python tools/run_experiment.py --experiment-config '${CONTAINER_EXPERIMENT_CONFIG}' synthesize"

    log "Stage 4b: Pseudo-label synthesized images in ${PSEUDO_SERVICE}"
    docker_exec_repo "${PSEUDO_SERVICE}" "python tools/run_experiment.py --experiment-config '${CONTAINER_EXPERIMENT_CONFIG}' obtain-pseudo"

    log "Stage 5: Train YOLO baseline and naive_v0 models in ${BASE_SERVICE}"
    docker_exec_repo "${BASE_SERVICE}" "python tools/run_experiment.py --experiment-config '${CONTAINER_EXPERIMENT_CONFIG}' train-yolo"

    log "Stage 6: Construct preference dataset in ${BASE_SERVICE}"
    docker_exec_repo "${BASE_SERVICE}" "python tools/run_experiment.py --experiment-config '${CONTAINER_EXPERIMENT_CONFIG}' construct-preference"

    log "Stage 7: Train DPO rephraser in ${DPO_SERVICE}"
    docker_exec_repo "${DPO_SERVICE}" "python tools/run_experiment.py --experiment-config '${CONTAINER_EXPERIMENT_CONFIG}' train-dpo"

    log "Updating config with latest automatic rephraser checkpoint"
    local automatic_ckpt
    automatic_ckpt="$(find_latest_checkpoint_dir "${AUTOMATIC_REPHRASER_GLOB}")"
    update_config_value "checkpoints.automatic_rephraser" "${automatic_ckpt}"

    if [[ "${RUN_AUTOMATIC_V1}" == "1" ]]; then
        log "Stage 8: Generate automatic_v1 captions in ${BASE_SERVICE}"
        docker_exec_repo "${BASE_SERVICE}" "python tools/run_experiment.py --experiment-config '${CONTAINER_EXPERIMENT_CONFIG}' extract-and-rephrase --include-automatic-v1"

        log "Stage 8b: Synthesize automatic_v1 images in ${GEN_SERVICE}"
        docker_exec_repo "${GEN_SERVICE}" "python tools/run_experiment.py --experiment-config '${CONTAINER_EXPERIMENT_CONFIG}' synthesize --include-automatic-v1"

        log "Stage 8c: Pseudo-label automatic_v1 images in ${PSEUDO_SERVICE}"
        docker_exec_repo "${PSEUDO_SERVICE}" "python tools/run_experiment.py --experiment-config '${CONTAINER_EXPERIMENT_CONFIG}' obtain-pseudo --include-automatic-v1"

        log "Stage 8d: Train YOLO automatic_v1 model in ${BASE_SERVICE}"
        docker_exec_repo "${BASE_SERVICE}" "python tools/run_experiment.py --experiment-config '${CONTAINER_EXPERIMENT_CONFIG}' train-yolo --config fisheye8k_with_naive_v0+automatic_v1"
    fi

    log "Stage 9: Evaluate all configured YOLO runs in ${BASE_SERVICE}"
    docker_exec_repo "${BASE_SERVICE}" "python tools/run_experiment.py --experiment-config '${CONTAINER_EXPERIMENT_CONFIG}' eval"

    log "Completed full ATES experiment pipeline"
    log "Final config: ${EXPERIMENT_CONFIG}"
}


main "$@"
