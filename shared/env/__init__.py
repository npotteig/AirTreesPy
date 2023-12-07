import sys
sys.path.append('../')
from gymnasium.envs.registration import register
from shared.env.env import AirSimEnv

register(
    id='AirSimEnv-v0',
    entry_point='shared.env.env:AirSimEnv',
    max_episode_steps=500
)