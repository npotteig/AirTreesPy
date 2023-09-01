import airsim
import navigation.bt_nodes as bt_nodes
import py_trees
import numpy as np
import time
from env import *
import gymnasium as gym

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
    reach_goal = bt_nodes.ReachGoal()
    goal_loop = py_trees.composites.Sequence(name='GoalLoop', memory=True)
    root.add_children([reach_goal, goal_loop])
    
    move_to = bt_nodes.move_to()
    get_sg = bt_nodes.gen_sg(manager_policy=manager_policy)
    goal_loop.add_children([get_sg, move_to])
    
    return root, blackboard

if __name__ == '__main__':
    dt = 0.1 
    z = -30
    vel = 10
    vehicle_name = "Drone1"
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    pos1 = airsim.Vector3r(100, 0, -32)
    pose1 = airsim.Pose(position_val=pos1)
    pos2 = airsim.Vector3r(-100, 0, -32)
    pose2 = airsim.Pose(position_val=pos2)
    pos3 = airsim.Vector3r(0, 100, -32)
    pose3 = airsim.Pose(position_val=pos3)
    pos4 = airsim.Vector3r(0, -100, -32)
    pose4 = airsim.Pose(position_val=pos4)
    
    scaleY = airsim.Vector3r(1, 200, 5)
    scaleX = airsim.Vector3r(200, 1, 5)
    client.simSpawnObject('my_cube', 'Cube', pose1, scaleY)
    client.simSpawnObject('my_cube', 'Cube', pose2, scaleY)
    client.simSpawnObject('my_cube', 'Cube', pose3, scaleX)
    client.simSpawnObject('my_cube', 'Cube', pose4, scaleX)
    
    # client.reset()
    # client.enableApiControl(True, vehicle_name)
    # client.armDisarm(True, vehicle_name)
    # client.takeoffAsync(vehicle_name=vehicle_name).join()
    
    # # Fixed Z Altitude
    # client.moveToZAsync(-31, velocity=15).join()
    # client.moveToPositionAsync(8, -150, -31, 5).join()
    
    
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