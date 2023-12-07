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
--load_dir "./runs/pretrained/models" \
--load_replay_buffer "./runs/pretrained/replay_data/" \
--load_safety_dir "./runs/pretrained/safety" \
--seed ${SEED}
