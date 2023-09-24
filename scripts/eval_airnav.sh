REWARD_SHAPING=$1
GPU=$2
SEED=$3

CUDA_VISIBLE_DEVICES=${GPU} python eval.py \
--env_name "AirSimEnv-v0" \
--type_of_env "large" \
--reward_shaping ${REWARD_SHAPING} \
--algo higl \
--version "${REWARD_SHAPING}" \
--seed ${SEED} \
--landmark_sampling fps \
--n_landmark_coverage 100 \
--delta 0.5 \
--adj_factor 0.7 \
--load \
--load_dir "./navigation/safe_fast/models" \
--load_replay_buffer "./navigation/replay_data/" \
--seed ${SEED}
