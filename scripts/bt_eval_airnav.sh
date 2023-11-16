ENV_TYPE=$1
SAMPLING=$2
BT_TYPE=$3
GPU=$4
SEED=$5

CUDA_VISIBLE_DEVICES=${GPU} python bt_eval.py \
--env_name "AirSimEnv-v0" \
--type_of_env ${ENV_TYPE} \
--bt_type ${BT_TYPE} \
--algo higl \
--version "dense" \
--seed ${SEED} \
--landmark_sampling ${SAMPLING} \
--n_landmark_coverage 100 \
--delta 0.5 \
--adj_factor 0.7 \
--load \
--load_dir "./navigation/paper_data/safe_layer/run0/models" \
--load_replay_buffer "./navigation/paper_data/safe_layer/run0/replay_data/" \
--load_safety_dir "./navigation/safety" \
--seed ${SEED}
