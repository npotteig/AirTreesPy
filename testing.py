import airsim
import navigation.bt_nodes_simple as bt_nodes_simple
import py_trees
import numpy as np
import time
from env import *
import gymnasium as gym
import airmap.airmap_objects as airobjects

class Manager():
    def __init__(self) -> None:
        self.subgoals = [np.array([20, 20, -30]),
                         np.array([40, 25, -30]),
                         np.array([125, 111, -30]),
                         np.array([200, 200, -30])]
        self.index = 0
    
    def sample_goal(self, state, goal):
        new_sg = self.subgoals[self.index]
        self.index += 1
        return new_sg

def build_tree(manager_policy):
    blackboard = py_trees.blackboard.Client(name="Global")
    blackboard.register_key(key="env", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="subgoal", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key='success', access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="state", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="done", access=py_trees.common.Access.WRITE)
    
    root = py_trees.composites.Selector(name='AirTree', memory=False)
    reach_goal = bt_nodes_simple.ReachGoal()
    goal_loop = py_trees.composites.Sequence(name='GoalLoop', memory=True)
    root.add_children([reach_goal, goal_loop])
    
    move_to = bt_nodes_simple.move_to()
    get_sg = bt_nodes_simple.gen_sg(manager_policy=manager_policy)
    goal_loop.add_children([get_sg, move_to])
    
    return root, blackboard

if __name__ == '__main__':
    dt = 0.1 
    z = -30
    vel = 10
    vehicle_name = "Drone1"
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    airobjects.destroy_walls(client)
    airobjects.spawn_walls(client, -100, 100, -32)
    airobjects.spawn_obstacles(client, -32)
    
    
    # while True:
    #     data_distance0 = client.getDistanceSensorData(distance_sensor_name="Distance0", vehicle_name="Drone1")
    #     sensor_read = 0
    #     if data_distance0.distance <= 10:
    #         sensor_read = (10 - data_distance0.distance) / 10
    #     state = client.simGetGroundTruthKinematics(vehicle_name)
    #     pos = state.position
    #     print(pos)
    #     time.sleep(1)
    client.reset()
    client.enableApiControl(True, vehicle_name)
    client.armDisarm(True, vehicle_name)
    client.takeoffAsync(vehicle_name=vehicle_name).join()
    
    client.moveToZAsync(-35, velocity=15).join()
    client.moveToPositionAsync(65, 80, -35, velocity=5).join()
    
    
    # manager = Manager()
    
    # root, blackboard = build_tree(manager)
    
    # env = gym.make('AirSimEnv-v0', client=client, dt=dt, vehicle_name=vehicle_name)
    # blackboard.env = env
    
    
    # blackboard.state, _ = blackboard.env.reset()    
        
    # blackboard.done = False
    # blackboard.success = False
    # tick = 0
    # while not blackboard.done:
        
    #     root.tick_once()
    #     print('Tick', tick)
    #     print(blackboard.state)
    #     # print("\n")
    #     # print(py_trees.display.unicode_tree(root=root, show_status=True))
    #     tick += 1
    #     time.sleep(dt)