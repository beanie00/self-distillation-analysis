nohup bash -c '
for model in \
beanie00/math-GRPO-Qwen3-8B-think-step-100 \
beanie00/math-SDPO-Qwen3-8B-think-step-100 \
beanie00/math-SDPO-no-think-contents-Qwen3-8B-think-step-100 \
beanie00/math-GRPO-DeepSeek-Distill-7B-step-100 \
beanie00/math-SDPO-no-think-contents-DeepSeek-Distill-7B-think-step-100 \
beanie00/math-SDPO-DeepSeek-Distill-7B-think-step-100 \
beanie00/math-GRPO-Qwen3-8B-think-off-step-100 \
beanie00/math-SDPO-Qwen3-8B-think-off-step-100
do

    # default
    temperature=0.6
    top_p=0.95

    # override for think-off
    if [[ "$model" == *"think-off"* ]]; then
        temperature=0.7
        top_p=0.8
    fi

    echo "Running $model | temp=$temperature top_p=$top_p"

    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py \
        --model_name_or_path ${model} \
        --data_path ../data/math/evaluation/aime24.parquet \
        --max_tokens 38912 \
        --enable_thinking \
        --temperature $temperature \
        --top_p $top_p \
        --n_sampling 4 \
        --k 4

done
' > eval_examples.log 2>&1 &