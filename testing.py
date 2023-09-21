import airsim
import navigation.bt_nodes_simple as bt_nodes_simple
import py_trees
import numpy as np
import time
from env import *
from env.env import AirWrapperEnv
import gymnasium as gym
import airmap.airmap_objects as airobjects
from airmap.blocks_tree_generator import build_blocks_world
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
    build_blocks_world(client=client, load=True)
    while True:
        print(client.simGetGroundTruthKinematics(vehicle_name).position)
    # airobjects.destroy_objects(client)
    # airobjects.spawn_walls(client, -100, 100, -32)
    # airobjects.spawn_obstacles(client, -32)
    # obs, info = env.reset()
    # client.moveByVelocityAsync(5, 5, 5, duration=10, drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(is_rate=False)).join()
    

    
    
    
    
