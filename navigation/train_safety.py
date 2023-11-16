import airsim
import pandas as pd
import os

import gymnasium as gym
from env import *
import numpy as np

from higl.safety_layer import SafetyLayer
from env.env import AirWrapperEnv

import airmap.airmap_objects as airobjects
from airmap.blocks_tree_generator import build_blocks_world

def run(args):
    if not os.path.exists("./navigation/safety"):
        os.makedirs("./navigation/safety")
    
    if args.load_buffer:
        if args.load_models:
            safelayer = SafetyLayer(device="cuda", load_buffer=args.load_replay_buffer, load_ckpt_dir=args.load_models_dir)
        else:
            safelayer = SafetyLayer(device="cuda", load_buffer=args.load_replay_buffer, load_ckpt_dir=args.load_models_dir)
    else:
        dt = 0.1 
        vehicle_name = "Drone1"
        client = airsim.MultirotorClient()
        client.confirmConnection()
        
        if args.type_of_env == "training":
            airobjects.destroy_objects(client)
            airobjects.spawn_walls(client, -100, 100, -32)
            airobjects.spawn_obstacles(client, -32)
        else:
            build_blocks_world(client=client, load=True)
        
        env = AirWrapperEnv(gym.make(args.env_name, client=client, dt=dt, vehicle_name=vehicle_name, type_of_env=args.type_of_env))
        
        if args.load_models:
            safelayer = SafetyLayer(env, device="cuda", load_ckpt_dir=args.load_models_dir)
        else:
            safelayer = SafetyLayer(env, device="cuda", load_ckpt_dir=args.load_models_dir)
    safelayer.train(batch_size=args.batch_size, lr=args.lr, sample_data_episodes=args.sample_data_episodes,
                    buffer_size=args.buffer_size, epochs=args.epochs, train_per_epoch=args.training_steps_per_epoch,
                    eval_per_epoch=args.evaluation_steps_per_epoch)
    if args.save_models or args.save_buffer:
        safelayer.save(args.save_dir, args.save_models, args.save_buffer)
