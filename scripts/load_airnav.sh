REWARD_SHAPING=$1
TIMESTEPS=$2
GPU=$3
SEED=$4

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
--env_name "AirSimEnv-v0" \
--reward_shaping ${REWARD_SHAPING} \
--algo higl \
--version "${REWARD_SHAPING}" \
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
--load_dir "./navigation/safe_fast/models" \
--save_models \
--save_replay_buffer "navigation/replay_data/" \
--seed ${SEED}