CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_with_hint.py \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --collected_path outputs/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/dapo_sample_8_100/train_think_n8_correct1-4_collect100.jsonl \
    --data_name math \
    --n_sample 100 \
    --n_sampling 8 \
    --temperature 0.6