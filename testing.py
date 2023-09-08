import airsim
import navigation.bt_nodes_simple as bt_nodes_simple
import py_trees
import numpy as np
import time
from env import *
from env.env import AirWrapperEnv
import gymnasium as gym
import airmap.airmap_objects as airobjects
import math

def calc_potential(sensor_info):
    # magnitudes = [2, 1, 3, 0, 4, 7, 5, 6]
    magnitudes = [0, 1, 7, 2, 6, 3, 5, 4]
    resultant_x = sensor_info[0] * math.cos(math.radians(magnitudes[0] * 45))
    resultant_y = sensor_info[1] * math.cos(math.radians(magnitudes[0] * 45))
    
    for i in range(1, 8):
        angle = math.radians(magnitudes[i] * 45)
        mag = sensor_info[i]
        
        x_component = mag * math.cos(angle)
        y_component = mag * math.sin(angle)
        
        resultant_x += x_component
        resultant_y += y_component
    
    res_array = np.array([resultant_x, resultant_y])
    # res_vec = res_array / np.linalg.norm(res_array)
    
    return -res_array 
        

if __name__ == '__main__':
    dt = 0.1 
    z = -30
    vel = 10
    vehicle_name = "Drone1"
    client = airsim.MultirotorClient()
    env = AirWrapperEnv(gym.make("AirSimEnv-v0", client=client, dt=dt, vehicle_name=vehicle_name))
    
    client.confirmConnection()
    
    airobjects.destroy_objects(client)
    airobjects.spawn_walls(client, -100, 100, -32)
    airobjects.spawn_obstacles(client, -32)
    
    env.evaluate = True
    obs, info = env.reset()
    state = obs['observation']
    done = False
    intervene = False
    actions_before_intervention = []
    intervene_index = 1
    thresh = 0.8
    while not done:
        if not intervene:
            action = np.array([0, 5])
            actions_before_intervention.append(state)
        else:
            potential = calc_potential(state[4:12])
            action = np.clip(5 * potential, -5, 5) 
            print(action)
            # action = [0, 0]
            # pos = actions_before_intervention[int(-1 * intervene_index)]
            # client.moveToPositionAsync(pos[0]*10, pos[1]*10, -30, 5, vehicle_name=vehicle_name)
            intervene_index += 1
            
        action_copy = action.copy()
        # if np.any(obs['observation'][4:12] > 0):
        #     action_copy = np.clip(action_copy, -3, 3)
        # print(action_copy)
        obs, rew, done, _, _ = env.step(action_copy)
        
        next_state = obs['observation']
        state = next_state
        inter_temp = False
        if np.any(next_state[4:12] > 0.7):
            inter_temp = True
        
        
        if inter_temp:
            intervene = True
        elif not inter_temp and intervene_index > 1:
            intervene = False
            actions_before_intervention = []
            intervene_index = 1
        
        print(obs['observation'][4:12])
    
    
    
    
