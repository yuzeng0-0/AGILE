ray stop
set -x

SAVE_CHECKPOINT_DIR=checkpoints/AGILE_grpo_15.6k_7B

# REF_MODEL_PATH: cold start model path
REF_MODEL_PATH=LLaMA-Factory/ckpt/qwen2_5vl_full_cold_start
PROJECT_NAME="AGILE"
EXPERIMENT_NAME="AGILE_grpo_15.6k_7B"

wandb login "Your WANDB_API_KEY"

TRAIN_DATA_DIR1=/path/to/dataset/AGILE/RL
TRAIN_DATA1=${TRAIN_DATA_DIR1}/BLINK_1523.parquet
TRAIN_DATA2=${TRAIN_DATA_DIR1}/COCO_1657.parquet
TRAIN_DATA3=${TRAIN_DATA_DIR1}/deepeyes_vstar_4473.parquet
TRAIN_DATA4=${TRAIN_DATA_DIR1}/HRBench4K_780.parquet
TRAIN_DATA5=${TRAIN_DATA_DIR1}/HRBench8K_796.parquet
TRAIN_DATA6=${TRAIN_DATA_DIR1}/POPE_1026.parquet
TRAIN_DATA7=${TRAIN_DATA_DIR1}/MMStar_1042.parquet
TRAIN_DATA8=${TRAIN_DATA_DIR1}/OCRBench_415.parquet
TRAIN_DATA9=${TRAIN_DATA_DIR1}/OCRVQA_1011.parquet
TRAIN_DATA10=${TRAIN_DATA_DIR1}/SEEDBench2_Plus_1139.parquet
TRAIN_DATA11=${TRAIN_DATA_DIR1}/TextVQA_VAL_1574.parquet
TRAIN_DATA12=${TRAIN_DATA_DIR1}/Vstar_189.parquet

TEST_DATA1=${TRAIN_DATA_DIR1}/Vstar_189.parquet

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +debug=false \
    +vs_debug=false \
    data.train_files=[${TRAIN_DATA1},${TRAIN_DATA2},${TRAIN_DATA3},${TRAIN_DATA4},${TRAIN_DATA5},${TRAIN_DATA6},${TRAIN_DATA7},${TRAIN_DATA8},${TRAIN_DATA9},${TRAIN_DATA10},${TRAIN_DATA11},${TRAIN_DATA12}] \
    data.val_files=[${TEST_DATA1}] \
    data.train_batch_size=64 \
    data.val_batch_size=512 \
    data.max_prompt_length=8192 \
    data.max_response_length=20000 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    +worker.actor.fsdp.torch_dtype=bf16 \
    +worker.actor.optim.strategy=adamw_bf16 \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=25000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=2048 \
    actor_rollout_ref.rollout.agent.max_turns=5 \
    actor_rollout_ref.rollout.agent.concurrent_workers=1 \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=20 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.total_epochs=1 2>&1 | tee ./outputs/${EXPERIMENT_NAME}/console.log