#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set these to your own paths
CKPT_DIR=""
EXPERIMENT=""

for step in 10 20 30 40 50 60 70 80 90 100; do
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir ${CKPT_DIR}/${EXPERIMENT}/global_step_${step}/actor \
        --target_dir ${CKPT_DIR}/${EXPERIMENT}/global_step_${step}/output_hf_model \
        --use_cpu_initialization
done

for step in 10 20 30 40 50 60 70 80 90; do
    rm -rf ${CKPT_DIR}/${EXPERIMENT}/global_step_${step}/actor
done