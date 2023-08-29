import py_trees
import numpy as np

class move_to(py_trees.behaviour.Behaviour):
    
    def __init__(self, name: str = "MoveTo"):
        super().__init__(name=name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key="client", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="vehicle_name", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="subgoal", access=py_trees.common.Access.READ)
        
    def setup(self):
        pass
    
    def initialise(self):
        self.call_move = True

    def update(self) -> py_trees.common.Status:
        pos = self.blackboard.client.simGetGroundTruthEnvironment(self.blackboard.vehicle_name).position
        pos_np = np.array([pos.x_val, pos.y_val, pos.z_val])
        if np.linalg.norm(pos_np - self.blackboard.subgoal) < 1:
            new_status = py_trees.common.Status.SUCCESS
        else:
            if self.call_move:
                self.blackboard.client.moveToPositionAsync(float(self.blackboard.subgoal[0]), 
                                                           float(self.blackboard.subgoal[1]), 
                                                           float(self.blackboard.subgoal[2]), 
                                                           velocity=1)
                self.call_move = False
            new_status = py_trees.common.Status.RUNNING
        return new_status

    def terminate(self, new_status: py_trees.common.Status) -> None:
        pass