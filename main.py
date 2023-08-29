import airsim
import tree.bt_nodes as bt_nodes
import py_trees
import numpy as np
import time
from env import *
import gymnasium as gym


def build_tree():
    blackboard = py_trees.blackboard.Client(name="Global")
    blackboard.register_key(key="env", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="subgoal", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="state", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="done", access=py_trees.common.Access.WRITE)
    
    root = bt_nodes.move_to()
    return root, blackboard

if __name__ == '__main__':
    dt = 1
    z = -30
    vel = 5
    vehicle_name = "Drone1"
    goal = np.array([8, 8, z])
    client = airsim.MultirotorClient()
    
    root, blackboard = build_tree()
    blackboard.subgoal = goal
    
    env = gym.make('AirSimEnv-v0', client=client, vel=vel, vehicle_name=vehicle_name, goal=goal)
    blackboard.env = env
    
    blackboard.state, _ = blackboard.env.reset()
    blackboard.done = False
    t = 0
    while not blackboard.done:
        print(t)
        root.tick_once()
        time.sleep(dt)
        t += 1