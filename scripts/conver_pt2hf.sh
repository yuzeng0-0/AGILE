export CUDA_VISIBLE_DEVICES=0

python3 scripts/model_merger.py \
  --backend fsdp \
  --hf_model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --local_dir checkpoints/AGILE_grpo_15.6k_7B/AGILE/AGILE_grpo_15.6k_7B/global_step_244/actor \
  --target_dir checkpoints/AGILE_grpo_15.6k_7B/AGILE/AGILE_grpo_15.6k_7B/global_step_244/actor/huggingface