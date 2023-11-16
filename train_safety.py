import argparse

from navigation.train_safety import run

parser = argparse.ArgumentParser()

# Environment
parser.add_argument("--env_name", required=True, type=str)
parser.add_argument("--type_of_env", default="training", type=str, choices=["training", "transfer"])

# Model Specific
parser.add_argument("--sample_data_episodes", default=2000., type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--epochs", default=512, type=int)
parser.add_argument("--buffer_size", default=1000, type=int)
parser.add_argument("--training_steps_per_epoch", default=1000, type=int)
parser.add_argument("--evaluation_steps_per_epoch", default=20, type=int)
# parser.add_argument("--safety_correction_scale", default=1.0, type=float)


# Save
parser.add_argument("--save_models", action="store_true")
parser.add_argument("--save_buffer", action="store_true")
parser.add_argument("--save_dir", default="./navigation/safety", type=str)


# Load
parser.add_argument("--load_buffer", action="store_true")
parser.add_argument("--load_replay_buffer", default="./navigation/safety", type=str)
parser.add_argument("--load_models", action="store_true")
parser.add_argument("--load_models_dir", default="./navigation/safety", type=str)


args = parser.parse_args()

# Run the algorithm
run(args)