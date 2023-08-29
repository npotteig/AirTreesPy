import sys
sys.path.append('../')
from gymnasium.envs.registration import register
from env.env import AirSimEnv

register(
    id='AirSimEnv-v0',
    entry_point='env.env:AirSimEnv',
    max_episode_steps=100
)