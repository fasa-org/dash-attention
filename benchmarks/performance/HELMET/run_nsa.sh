MODEL_PATH="/yourpath"
OUTPUT_ROOT="./results_nsa_8B_sft"

TASKS=(
    "helmet_icl__16384::suite"
    "helmet_longqa__16384::suite"
    "helmet_rag__16384::suite"
    "helmet_recall__16384::suite"
    "helmet_rerank__16384::suite"
    "helmet_summ__16384::suite"
    "helmet_cite__16384::suite"
)

rm -rf ${OUTPUT_ROOT}/

for i in ${!TASKS[@]}
do
    TASK=${TASKS[$i]}
    GPU=$i

    echo "Running $TASK on GPU $GPU"

    CUDA_VISIBLE_DEVICES=$GPU olmes \
        --model $MODEL_PATH \
        --model-type hf \
        --batch-size 1 \
        --limit 100 \
        --task "$TASK" \
        --model-args '{"trust_remote_code": true, "add_bos_token": true}' \
        --output-dir ${OUTPUT_ROOT}/${TASK} &

done
wait