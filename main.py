import airsim
import tree.bt_nodes as bt_nodes
import py_trees
import numpy as np
import time


def build_tree():
    blackboard = py_trees.blackboard.Client(name="Global")
    blackboard.register_key(key="client", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="vehicle_name", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="subgoal", access=py_trees.common.Access.WRITE)
    root = bt_nodes.move_to()
    return root, blackboard

if __name__ == '__main__':
    dt = 1
    root, blackboard = build_tree()
    
    blackboard.vehicle_name = "Drone1"
    blackboard.subgoal = np.array([8, 8, -10])
    blackboard.client = airsim.MultirotorClient()
    blackboard.client.confirmConnection()
    blackboard.client.enableApiControl(True, blackboard.vehicle_name)
    blackboard.client.armDisarm(True, blackboard.vehicle_name)
    
    airsim.wait_key("Press any key to takeoff")
    f = blackboard.client.takeoffAsync(vehicle_name=blackboard.vehicle_name)
    f.join()
    
    t = 0
    while t < 100:
        root.tick_once()
        time.sleep(dt)