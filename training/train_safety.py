import airsim
import pandas as pd
import os
import torch

import gymnasium as gym
from shared.env import *
import numpy as np

from shared.higl.safety_layer import SafetyLayer
from shared.env.env import AirWrapperEnv
from shared.world_map import BlocksMaze, BlocksTrees 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(args):
    if not os.path.exists("./runs/safety"):
        os.makedirs("./runs/safety")
    
    if args.load_buffer:
        if args.load_models:
            safelayer = SafetyLayer(device=device, load_buffer=args.load_replay_buffer, load_ckpt_dir=args.load_models_dir)
        else:
            safelayer = SafetyLayer(device=device, load_buffer=args.load_replay_buffer)
    else:
        dt = 0.1 
        vehicle_name = "Drone1"
        client = airsim.MultirotorClient()
        client.confirmConnection()
        
        if args.type_of_env == "training":
            wrld_map = BlocksMaze(client)
        else:
            wrld_map = BlocksTrees(client, load=True)
        wrld_map.build_world()
        
        env = AirWrapperEnv(gym.make(args.env_name, client=client, dt=dt, vehicle_name=vehicle_name, type_of_env=args.type_of_env, randomize_start=True))
        
        if args.load_models:
            safelayer = SafetyLayer(env, device=device, load_ckpt_dir=args.load_models_dir)
        else:
            safelayer = SafetyLayer(env, device=device)
    safelayer.train(batch_size=args.batch_size, lr=args.lr, sample_data_episodes=args.sample_data_episodes,
                    buffer_size=args.buffer_size, epochs=args.epochs, train_per_epoch=args.training_steps_per_epoch,
                    eval_per_epoch=args.evaluation_steps_per_epoch)
    if args.save_models or args.save_buffer:
        safelayer.save(args.save_dir, args.save_models, args.save_buffer)
