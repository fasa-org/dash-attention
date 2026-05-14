#!/usr/bin/env bash

# Run all 13 RULER synthetic tasks with at most one task per GPU.
# Usage: bash run_ruler_13_tasks_8gpus.sh <model_name> [benchmark]

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <model_name> [benchmark]"
    exit 1
fi

MODEL_NAME=$1
BENCHMARK=${2:-synthetic}

if [ "${BENCHMARK}" != "synthetic" ]; then
    echo "Only benchmark 'synthetic' is supported by this launcher."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1

# Keep the same defaults as run.sh.
GPUS="1"
ROOT_DIR="/home/test/test01/hyx/sparse_train/sparse_train/test/RULER/generated"
MODEL_DIR="/home/test/test01/hyx"
ENGINE_DIR="."
BATCH_SIZE=1

source config_models.sh
source config_tasks.sh

MODEL_CONFIG=$(MODEL_SELECT "${MODEL_NAME}" "${MODEL_DIR}" "${ENGINE_DIR}")
IFS=":" read -r MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "${MODEL_CONFIG}"
if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi

export OPENAI_API_KEY=${OPENAI_API_KEY}
export GEMINI_API_KEY=${GEMINI_API_KEY}
export AZURE_API_ID=${AZURE_ID}
export AZURE_API_SECRET=${AZURE_SECRET}
export AZURE_API_ENDPOINT=${AZURE_ENDPOINT}

TASKS=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)

GPU_IDS=(0 1 2 3 4 5 6 7)
BASE_PORT=${BASE_PORT:-5000}
RUN_ID=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/logs/parallel_${RUN_ID}"
mkdir -p "${LOG_DIR}"

pids=()
pid_tasks=()
pid_gpus=()
gpu_busy=()
for _ in "${GPU_IDS[@]}"; do
    gpu_busy+=(0)
done

wait_for_http_server() {
    local port=$1
    local tries=${SERVER_WAIT_TRIES:-120}
    local i

    for ((i = 1; i <= tries; i++)); do
        if python - "${port}" >/dev/null 2>&1 <<'PY'
import sys
import urllib.request

port = sys.argv[1]
try:
    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
except Exception:
    raise SystemExit(1)
PY
        then
            return 0
        fi
        sleep 2
    done

    echo "Timed out waiting for server on port ${port}"
    return 1
}

start_server_if_needed() {
    local port=$1
    STARTED_SERVER_PID=""

    if [ "${MODEL_FRAMEWORK}" = "vllm" ]; then
        python pred/serve_vllm.py \
            --model="${MODEL_PATH}" \
            --tensor-parallel-size="${GPUS}" \
            --dtype bfloat16 \
            --disable-custom-all-reduce \
            --port "${port}" &
        STARTED_SERVER_PID=$!
        wait_for_http_server "${port}"
    elif [ "${MODEL_FRAMEWORK}" = "trtllm" ]; then
        python pred/serve_trt.py \
            --model_path="${MODEL_PATH}" \
            --port "${port}" &
        STARTED_SERVER_PID=$!
        sleep "${TRT_SERVER_STARTUP_SLEEP:-60}"
    elif [ "${MODEL_FRAMEWORK}" = "sglang" ]; then
        python -m sglang.launch_server \
            --model-path "${MODEL_PATH}" \
            --tp "${GPUS}" \
            --port "${port}" \
            --enable-flashinfer &
        STARTED_SERVER_PID=$!
        wait_for_http_server "${port}"
    fi
}

run_one_task() {
    local task=$1
    local gpu=$2
    local port=$3
    local total_time=0
    local server_pid=""

    export CUDA_VISIBLE_DEVICES="${gpu}"
    echo "[$(date '+%F %T')] START task=${task} gpu=${gpu} port=${port} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

    start_server_if_needed "${port}"
    server_pid="${STARTED_SERVER_PID}"
    if [ -n "${server_pid}" ]; then
        trap 'kill "${server_pid}" >/dev/null 2>&1 || true' EXIT
    fi

    for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
        RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
        DATA_DIR="${RESULTS_DIR}/data"
        PRED_DIR="${RESULTS_DIR}/pred"
        mkdir -p "${DATA_DIR}" "${PRED_DIR}"

        python data/prepare.py \
            --save_dir "${DATA_DIR}" \
            --benchmark "${BENCHMARK}" \
            --task "${task}" \
            --tokenizer_path "${TOKENIZER_PATH}" \
            --tokenizer_type "${TOKENIZER_TYPE}" \
            --max_seq_length "${MAX_SEQ_LENGTH}" \
            --model_template_type "${MODEL_TEMPLATE_TYPE}" \
            --num_samples "${NUM_SAMPLES}" \
            ${REMOVE_NEWLINE_TAB}

        start_time=$(date +%s)
        python pred/call_api.py \
            --data_dir "${DATA_DIR}" \
            --save_dir "${PRED_DIR}" \
            --benchmark "${BENCHMARK}" \
            --task "${task}" \
            --server_type "${MODEL_FRAMEWORK}" \
            --server_port "${port}" \
            --model_name_or_path "${MODEL_PATH}" \
            --temperature "${TEMPERATURE}" \
            --top_k "${TOP_K}" \
            --top_p "${TOP_P}" \
            --batch_size "${BATCH_SIZE}" \
            ${STOP_WORDS}
        end_time=$(date +%s)
        total_time=$((total_time + end_time - start_time))
    done

    if [ -n "${server_pid}" ]; then
        kill "${server_pid}" >/dev/null 2>&1 || true
        trap - EXIT
    fi

    echo "[$(date '+%F %T')] DONE task=${task} gpu=${gpu} call_api_seconds=${total_time}"
}

reap_finished_jobs() {
    local i pid status failed=0

    for ((i = 0; i < ${#pids[@]}; i++)); do
        pid=${pids[$i]}
        if [ "${pid}" = "" ]; then
            continue
        fi

        if ! kill -0 "${pid}" >/dev/null 2>&1; then
            wait "${pid}"
            status=$?
            echo "Finished task=${pid_tasks[$i]} gpu=${pid_gpus[$i]} status=${status}"
            gpu_busy[${pid_gpus[$i]}]=0
            pids[$i]=""
            if [ "${status}" -ne 0 ]; then
                failed=1
            fi
        fi
    done

    return "${failed}"
}

get_free_gpu() {
    local gpu
    for gpu in "${GPU_IDS[@]}"; do
        if [ "${gpu_busy[$gpu]}" -eq 0 ]; then
            echo "${gpu}"
            return 0
        fi
    done
    return 1
}

echo "Launching ${#TASKS[@]} tasks on GPUs: ${GPU_IDS[*]}"
echo "Logs: ${LOG_DIR}"

overall_status=0
for task in "${TASKS[@]}"; do
    while ! gpu=$(get_free_gpu); do
        if ! reap_finished_jobs; then
            overall_status=1
        fi
        sleep 10
    done

    port=$((BASE_PORT + gpu))
    log_file="${LOG_DIR}/${task}.gpu${gpu}.log"
    echo "Launch task=${task} gpu=${gpu} port=${port} log=${log_file}"
    run_one_task "${task}" "${gpu}" "${port}" > "${log_file}" 2>&1 &
    pid=$!
    pids+=("${pid}")
    pid_tasks+=("${task}")
    pid_gpus+=("${gpu}")
    gpu_busy[$gpu]=1
done

while true; do
    running=0
    for pid in "${pids[@]}"; do
        if [ -n "${pid}" ] && kill -0 "${pid}" >/dev/null 2>&1; then
            running=1
            break
        fi
    done

    if ! reap_finished_jobs; then
        overall_status=1
    fi

    if [ "${running}" -eq 0 ]; then
        break
    fi
    sleep 10
done

if [ "${overall_status}" -ne 0 ]; then
    echo "At least one task failed. Check logs in ${LOG_DIR}"
    exit "${overall_status}"
fi

for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    PRED_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}/pred"
    python eval/evaluate.py \
        --data_dir "${PRED_DIR}" \
        --benchmark "${BENCHMARK}"
done

echo "All tasks finished successfully."
