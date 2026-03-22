#!/bin/bash

# Set these to your own paths
CKPT_DIR=""
EXPERIMENT=""

nohup bash -c "
for step in 10 20 30 40 50 60 70 80 90 100; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py \
        --model_name_or_path ${CKPT_DIR}/${EXPERIMENT}/global_step_\${step}/output_hf_model \
        --data_path ../data/math/evaluation/aime24.parquet \
        --max_tokens 38912 \
        --enable_thinking \
        --temperature 0.6 \
        --top_p 0.95 \
        --n_sampling 16 \
        --k 16
done
" > eval_${EXPERIMENT}.log 2>&1 &