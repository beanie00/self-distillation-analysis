CUDA_VISIBLE_DEVICES=0,1,2,3 python make_sft_dataset.py \
    --input ./outputs_chained/chained_results_all.jsonl \
    --output_dir ./sft_datasets