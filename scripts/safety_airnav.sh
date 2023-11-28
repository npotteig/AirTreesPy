CUDA_VISIBLE_DEVICES=0 python train_safety.py \
--env_name "AirSimEnv-v0" \
--type_of_env "training" \
--sample_data_episodes 2000 \
--save_models \
--save_buffer \