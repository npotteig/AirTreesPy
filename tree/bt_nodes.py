import py_trees
import numpy as np

# Returns True if the Final Objective is reached
class ReachGoal(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "ReachGoal"):
        super().__init__(name=name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key='done', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='success', access=py_trees.common.Access.READ)
        self.blackboard.register_key(key='env', access=py_trees.common.Access.WRITE)

    def setup(self):
        pass

    def initialise(self):
        pass

    def update(self):
        if self.blackboard.success:
            self.blackboard.done = True
            new_status = py_trees.common.Status.SUCCESS
        else:
            new_status = py_trees.common.Status.FAILURE
        return new_status

    def terminate(self, new_status):
        pass

class move_to(py_trees.behaviour.Behaviour):
    
    def __init__(self, name: str = "MoveTo"):
        super().__init__(name=name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key="env", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="subgoal", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="state", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="success", access=py_trees.common.Access.WRITE)
        
    def setup(self):
        pass
    
    def initialise(self):
        pass

    def update(self) -> py_trees.common.Status:
        if np.linalg.norm(self.blackboard.state[:3] - self.blackboard.subgoal) < 2.5:
            new_status = py_trees.common.Status.SUCCESS
        else:
            new_status = py_trees.common.Status.RUNNING
        self.blackboard.state, rew, self.blackboard.success, _, _ = self.blackboard.env.step(self.blackboard.subgoal)
        return new_status

    def terminate(self, new_status: py_trees.common.Status) -> None:
        pass

class gen_sg(py_trees.behaviour.Behaviour):
    
    def __init__(self, manager_policy, name: str = "GenerateSubGoal"):
        super().__init__(name=name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key="env", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="subgoal", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="state", access=py_trees.common.Access.READ)
        self.manager_policy = manager_policy
        
    def setup(self):
        pass
    
    def initialise(self):
        pass

    def update(self) -> py_trees.common.Status:
        self.blackboard.subgoal = self.manager_policy.sample_goal(self.blackboard.state, self.blackboard.env.goal)
        new_status = py_trees.common.Status.SUCCESS
        return new_status

    def terminate(self, new_status: py_trees.common.Status) -> None:
        pass