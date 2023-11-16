TIMESTEPS=$1
GPU=$2
SEED=$3

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
--env_name "AirSimEnv-v0" \
--type_of_env "ansr" \
--algo higl \
--version "dense" \
--seed ${SEED} \
--max_timesteps ${TIMESTEPS} \
--landmark_sampling fps \
--n_landmark_coverage 20 \
--use_novelty_landmark \
--novelty_algo rnd \
--n_landmark_novelty 20 \
--delta 0.5 \
--adj_factor 0.7 \
--load \
--load_dir "./navigation/paper_data/safe_layer/run0/models" \
--load_safety_dir "./navigation/safety" \
--save_models \
--save_replay_buffer "navigation/replay_data/" \
--seed ${SEED}