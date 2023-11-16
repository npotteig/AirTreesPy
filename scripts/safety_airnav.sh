CUDA_VISIBLE_DEVICES=${GPU} python train_safety.py \
--env_name "AirSimEnv-v0" \
--type_of_env "small" \
--save_models \
--save_buffer \