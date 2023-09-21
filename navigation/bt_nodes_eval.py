import py_trees
import numpy as np
import higl.utils as utils

# Returns True if the Final Objective is reached
class ReachGoal(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "ReachGoal?"):
        super().__init__(name=name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key='done', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='success', access=py_trees.common.Access.READ)
        self.blackboard.register_key(key='goals_achieved', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='env', access=py_trees.common.Access.WRITE)

    def setup(self):
        pass

    def initialise(self):
        pass

    def update(self):
        if self.blackboard.success:
            self.blackboard.done = True
            self.blackboard.goals_achieved += 1
            new_status = py_trees.common.Status.SUCCESS
        else:
            new_status = py_trees.common.Status.FAILURE
        return new_status

    def terminate(self, new_status):
        pass
    
class CloseToLd(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "CloseToLd?"):
        super().__init__(name=name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key='achieved_goal', access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="landmark", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="built_landmark_graph", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key='success', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='goal', access=py_trees.common.Access.READ)

    def setup(self):
        pass

    def initialise(self):
        pass

    def update(self):
        new_status = py_trees.common.Status.SUCCESS if self.blackboard.built_landmark_graph and -np.linalg.norm(self.blackboard.landmark - self.blackboard.achieved_goal, axis=-1) > -1.0 else py_trees.common.Status.FAILURE
        if np.all(self.blackboard.landmark == self.blackboard.goal) and new_status == py_trees.common.Status.SUCCESS:
            self.blackboard.success = True
            
        return new_status

    def terminate(self, new_status):
        pass
    
class RetrievedPotentialLds(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "RetrievedPotentialLds?"):
        super().__init__(name=name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key="built_landmark_graph", access=py_trees.common.Access.READ)

    def setup(self):
        pass

    def initialise(self):
        pass

    def update(self):
        new_status = py_trees.common.Status.SUCCESS if self.blackboard.built_landmark_graph else py_trees.common.Status.FAILURE
        return new_status

    def terminate(self, new_status):
        pass

class NotSafe(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "NotSafe?"):
        super().__init__(name=name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key='state', access=py_trees.common.Access.READ)
        self.blackboard.register_key(key='manager_propose_frequency', access=py_trees.common.Access.READ)

    def setup(self):
        pass

    def initialise(self):
        pass

    def update(self):
        if np.any(self.blackboard.state[4:12] > 0.80) :
            new_status = py_trees.common.Status.SUCCESS
        else:
            new_status = py_trees.common.Status.FAILURE
        return new_status

    def terminate(self, new_status):
        pass
    
class ComputePotentialLds(py_trees.behaviour.Behaviour):

    def __init__(self, manager_policy, controller_policy, controller_replay_buffer, name: str = "ComputePotentialLds!"):
        super().__init__(name=name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key="built_landmark_graph", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='goal', access=py_trees.common.Access.READ)

        self.manager_policy = manager_policy
        self.controller_policy = controller_policy
        self.controller_replay_buffer = controller_replay_buffer

    def setup(self):
        pass

    def initialise(self):
        pass

    def update(self):
        self.manager_policy.init_planner()
        self.manager_policy.planner.eval_build_landmark_graph(self.blackboard.goal, self.controller_policy, self.controller_replay_buffer)
        self.blackboard.built_landmark_graph = True
        new_status = py_trees.common.Status.SUCCESS
        return new_status

    def terminate(self, new_status):
        pass
    
class GetNextLd(py_trees.behaviour.Behaviour):

    def __init__(self, manager_policy, name: str = "GetNextLd!"):
        super().__init__(name=name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key="landmark", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="ld_idx", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='state', access=py_trees.common.Access.READ)
        self.blackboard.register_key(key='goal', access=py_trees.common.Access.READ)

        self.manager_policy = manager_policy

    def setup(self):
        pass

    def initialise(self):
        pass

    def update(self):
        if np.any(self.blackboard.landmark != self.blackboard.goal):
            self.blackboard.landmark, self.blackboard.ld_idx = self.manager_policy.planner.get_next_landmark(self.blackboard.state,
                                                                                                        self.blackboard.landmark,
                                                                                                        self.blackboard.goal,
                                                                                                        self.blackboard.ld_idx)
        new_status = py_trees.common.Status.SUCCESS
        print(self.blackboard.landmark)
        return new_status

    def terminate(self, new_status):
        pass

class move_to(py_trees.behaviour.Behaviour):
    
    def __init__(self, controller_policy, calculate_controller_reward, ctrl_rew_scale, name: str = "MoveToSubGoal!"):
        super().__init__(name=name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key='subgoal', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='goal', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='achieved_goal', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='state', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='manager_propose_frequency', access=py_trees.common.Access.READ)
        self.blackboard.register_key(key='env', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='success', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="reward", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="sg_move_count", access=py_trees.common.Access.WRITE)

        # Evaluation Metrics
        self.blackboard.register_key(key="avg_reward", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="avg_controller_rew", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="global_steps", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="step_count", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="collision_count", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="evaluation", access=py_trees.common.Access.WRITE)

        self.controller_policy = controller_policy
        self.calculate_controller_reward = calculate_controller_reward
        self.ctrl_rew_scale = ctrl_rew_scale

    def setup(self):
        pass

    def initialise(self):
        pass

    def update(self):
        if (self.blackboard.sg_move_count % (self.blackboard.manager_propose_frequency + 1) == 0) :
            new_status = py_trees.common.Status.SUCCESS
        else:
            action = self.controller_policy.select_action(self.blackboard.state, self.blackboard.subgoal)
            new_obs, self.blackboard.reward, done, _, info = self.blackboard.env.step(action)
            self.blackboard.success = info['is_success'] or done
                
            if new_obs['observation'][-1] == 1:
                self.blackboard.collision_count += 1

            self.blackboard.goal = new_obs["desired_goal"]
            new_achieved_goal = new_obs['achieved_goal']
            new_state = new_obs["observation"]
            self.blackboard.subgoal = self.controller_policy.subgoal_transition(self.blackboard.achieved_goal, self.blackboard.subgoal, new_achieved_goal)

            self.blackboard.avg_reward += self.blackboard.reward
            self.ccr = self.calculate_controller_reward(self.blackboard.achieved_goal, self.blackboard.subgoal, new_achieved_goal, self.ctrl_rew_scale)
            self.blackboard.avg_controller_rew += self.ccr

            self.blackboard.global_steps += 1
            self.blackboard.step_count += 1
            self.blackboard.sg_move_count += 1
            self.blackboard.state = new_state
            self.blackboard.achieved_goal = new_achieved_goal

            new_status = py_trees.common.Status.RUNNING
        return new_status

    def terminate(self, new_status):
        pass

class safe_move_to(py_trees.behaviour.Behaviour):
    
    def __init__(self, controller_policy, calculate_controller_reward, ctrl_rew_scale, name: str = "SafeMoveToSubGoal!"):
        super().__init__(name=name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key='subgoal', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='goal', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='achieved_goal', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='state', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='manager_propose_frequency', access=py_trees.common.Access.READ)
        self.blackboard.register_key(key='env', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='success', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="reward", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="sg_move_count", access=py_trees.common.Access.WRITE)

        # Evaluation Metrics
        self.blackboard.register_key(key="avg_reward", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="avg_controller_rew", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="global_steps", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="step_count", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="collision_count", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="evaluation", access=py_trees.common.Access.WRITE)

        self.controller_policy = controller_policy
        self.calculate_controller_reward = calculate_controller_reward
        self.ctrl_rew_scale = ctrl_rew_scale

    def setup(self):
        pass

    def initialise(self):
        pass

    def update(self):
        if (self.blackboard.sg_move_count % (self.blackboard.manager_propose_frequency + 1) == 0) :
            new_status = py_trees.common.Status.SUCCESS
        else:
            potential = utils.calc_potential(self.blackboard.state[4:12])
            action = np.clip(5 * potential, -5, 5)
            new_obs, self.blackboard.reward, done, _, info = self.blackboard.env.step(action)
            self.blackboard.success = info['is_success'] or done
                
            if new_obs['observation'][-1] == 1:
                self.blackboard.collision_count += 1

            self.blackboard.goal = new_obs["desired_goal"]
            new_achieved_goal = new_obs['achieved_goal']
            new_state = new_obs["observation"]
            self.blackboard.subgoal = self.controller_policy.subgoal_transition(self.blackboard.achieved_goal, self.blackboard.subgoal, new_achieved_goal)

            self.blackboard.avg_reward += self.blackboard.reward
            self.ccr = self.calculate_controller_reward(self.blackboard.achieved_goal, self.blackboard.subgoal, new_achieved_goal, self.ctrl_rew_scale)
            self.blackboard.avg_controller_rew += self.ccr

            self.blackboard.global_steps += 1
            self.blackboard.step_count += 1
            self.blackboard.sg_move_count += 1
            self.blackboard.state = new_state
            self.blackboard.achieved_goal = new_achieved_goal

            new_status = py_trees.common.Status.RUNNING
        return new_status

    def terminate(self, new_status):
        pass

class gen_sg(py_trees.behaviour.Behaviour):
    
    def __init__(self, manager_policy, name: str = "GenerateSubGoal!"):
        super().__init__(name=name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key="landmark", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="subgoal", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="state", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="sg_move_count", access=py_trees.common.Access.WRITE)
        self.manager_policy = manager_policy
        
    def setup(self):
        pass
    
    def initialise(self):
        pass

    def update(self) -> py_trees.common.Status:
        self.blackboard.subgoal = self.manager_policy.sample_goal(self.blackboard.state, self.blackboard.landmark)
        # if np.any(self.blackboard.state[4:12] > 0.80):
        #     potential = utils.calc_potential(self.blackboard.state[4:12])
        #     self.blackboard.subgoal += 1.0*potential
        self.blackboard.sg_move_count = 1
        new_status = py_trees.common.Status.SUCCESS
        return new_status

    def terminate(self, new_status: py_trees.common.Status) -> None:
        pass