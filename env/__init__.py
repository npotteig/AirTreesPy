import sys
sys.path.append('../')
from gymnasium.envs.registration import register
from env.env import AirSimEnv
from env.simple_env import SimpleEnv

register(
    id='AirSimEnv-v0',
    entry_point='env.env:AirSimEnv',
    max_episode_steps=500
)

register(
    id='SimpleEnv-v0',
    entry_point='env.simple_env:SimpleEnv',
    max_episode_steps=500
)