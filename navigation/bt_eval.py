import torch

import numpy as np

import higl.utils as utils
import higl.higl as higl

import airsim

import pandas as pd
import os

from higl.safety_layer import SafetyLayer

import gymnasium as gym
from env import *

from env.env import AirWrapperEnv

import airmap.airmap_objects as airobjects
from airmap.blocks_tree_generator import build_blocks_world
from airmap.blocks_tree_generator import obstacle_info

import navigation.bt_nodes_eval as bt_nodes
import py_trees

import time

def build_blackboard(environment, manager_propose_frequency=10):
    # Setup BT stuff here
    blackboard = py_trees.blackboard.Client(name="Global")
    blackboard.register_key(key="env", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="goal", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key='achieved_goal', access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="state", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="subgoal", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="done", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="success", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="reward", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="manager_propose_frequency", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="sg_move_count", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="built_landmark_graph", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="landmark", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="ld_idx", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="goal_list", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="goal_idx", access=py_trees.common.Access.WRITE)
    
    blackboard.register_key(key="constraints", access=py_trees.common.Access.WRITE)

    # Evaluation Metrics
    blackboard.register_key(key="avg_reward", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="avg_controller_rew", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="global_steps", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="evaluation", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="goals_achieved", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="step_count", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="env_goals_achieved", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="collision_count", access=py_trees.common.Access.WRITE)

    blackboard.env = environment
    blackboard.manager_propose_frequency = manager_propose_frequency
    
    return blackboard
    

def build_baseline_bt(manager_policy, 
             controller_policy,
             safelayer,
             controller_replay_buffer, 
             calculate_controller_reward, 
             ctrl_rew_scale):
    root = py_trees.composites.Sequence(name='AirTreeSeq', memory=True)
    potential_ld_fall = py_trees.composites.Selector(name='PotLdFall',memory=False)
    control_loop_fall = py_trees.composites.Selector(name='CtrlFall', memory=False)
    root.add_children([potential_ld_fall, control_loop_fall])
    
    retrieved_potential_ld = bt_nodes.RetrievedPotentialLds()
    compute_potential_ld = bt_nodes.ComputePotentialLds(manager_policy, controller_policy, controller_replay_buffer, 
                                                        step_size=2, obstacle_info=obstacle_info)
    potential_ld_fall.add_children([retrieved_potential_ld, compute_potential_ld])
    
    reach_goal = bt_nodes.ReachGoal()
    move_seq = py_trees.composites.Sequence(name='MoveSeq', memory=True)
    control_loop_fall.add_children([reach_goal, move_seq])
    
    sg_fall = py_trees.composites.Selector(name='SGFall', memory=False)
    move_to = bt_nodes.move_to(controller_policy, calculate_controller_reward, ctrl_rew_scale)
    move_seq.add_children([sg_fall, move_to])
    
    not_ready_sg = bt_nodes.NotReadyForSG()
    get_sg = bt_nodes.gen_sg(manager_policy)
    sg_fall.add_children([not_ready_sg, get_sg])
    
    return root
    

def build_expert_bt(manager_policy, 
             controller_policy,
             safelayer,
             controller_replay_buffer, 
             calculate_controller_reward, 
             ctrl_rew_scale):
    
    root = py_trees.composites.Sequence(name='AirTreeSeq', memory=True)
    potential_ld_fall = py_trees.composites.Selector(name='PotLdFall',memory=False)
    control_loop_fall = py_trees.composites.Selector(name='CtrlFall', memory=False)
    root.add_children([potential_ld_fall, control_loop_fall])
    
    retrieved_potential_ld = bt_nodes.RetrievedPotentialLds()
    compute_potential_ld = bt_nodes.ComputePotentialLds(manager_policy, controller_policy, controller_replay_buffer, 
                                                        step_size=2, obstacle_info=obstacle_info)
    potential_ld_fall.add_children([retrieved_potential_ld, compute_potential_ld])
    
    reach_goal = bt_nodes.ReachGoal()
    goal_loop = py_trees.composites.Selector(name='GoalLoopFall', memory=False)
    control_loop_fall.add_children([reach_goal, goal_loop])
    
    landmark_seq = py_trees.composites.Sequence(name='LdSeq', memory=True)
    move_seq = py_trees.composites.Sequence(name='MoveSeq', memory=True)
    goal_loop.add_children([landmark_seq, move_seq])
    
    close_to_ld = bt_nodes.CloseToLd()
    get_next_ld = bt_nodes.GetNextLd(manager_policy)
    landmark_seq.add_children([close_to_ld, get_next_ld])
    
    sg_fall = py_trees.composites.Selector(name='SGFall', memory=False)
    lowlevel_loop = py_trees.composites.Selector(name='MoveLoopFall', memory=False)
    move_seq.add_children([sg_fall, lowlevel_loop])
    
    not_ready_sg = bt_nodes.NotReadyForSG()
    get_sg = bt_nodes.gen_sg(manager_policy)
    sg_fall.add_children([not_ready_sg, get_sg])
    
    move_to = bt_nodes.move_to(controller_policy, calculate_controller_reward, ctrl_rew_scale)
    safe_seq = py_trees.composites.Sequence(name='SafeSeq', memory=False)
    lowlevel_loop.add_children([safe_seq, move_to])
    
    not_safe = bt_nodes.NotSafe()
    safe_move_to = bt_nodes.safe_move_to(controller_policy, safelayer, calculate_controller_reward, ctrl_rew_scale)
    safe_seq.add_children([not_safe, safe_move_to])
    
    return root

def build_goal_change_bt(manager_policy, 
             controller_policy,
             safelayer,
             controller_replay_buffer, 
             calculate_controller_reward, 
             ctrl_rew_scale):
    root = py_trees.composites.Selector(name='AirTreeFall', memory=False)
    reach_final_goal = bt_nodes.ReachGoal(name="ReachFinalGoal?")
    execution_seq = py_trees.composites.Sequence(name='ExecSeq', memory=False)
    root.add_children([reach_final_goal, execution_seq])
    
    goal_change_fall = py_trees.composites.Selector(name='GoalChgFall',memory=False)
    goal_loop = py_trees.composites.Selector(name='GoalLoopFall', memory=False)
    execution_seq.add_children([goal_change_fall, goal_loop])
    
    not_need_new_goal = bt_nodes.NotChangeGoal()
    chg_goal_seq = py_trees.composites.Sequence(name='ChgGoalSeq', memory=False)
    goal_change_fall.add_children([not_need_new_goal, chg_goal_seq])
    
    chg_goal = bt_nodes.UpdateGoal()
    compute_potential_ld = bt_nodes.ComputePotentialLds(manager_policy, controller_policy, controller_replay_buffer, 
                                                        step_size=2, obstacle_info=obstacle_info)
    chg_goal_seq.add_children([chg_goal, compute_potential_ld])
    
    landmark_seq = py_trees.composites.Sequence(name='LdSeq', memory=False)
    move_seq = py_trees.composites.Sequence(name='MoveSeq', memory=False)
    goal_loop.add_children([landmark_seq, move_seq])
    
    close_to_ld = bt_nodes.CloseToLd()
    get_next_ld = bt_nodes.GetNextLd(manager_policy)
    landmark_seq.add_children([close_to_ld, get_next_ld])
    
    sg_fall = py_trees.composites.Selector(name='SGFall', memory=False)
    lowlevel_loop = py_trees.composites.Selector(name='MoveLoopFall', memory=False)
    move_seq.add_children([sg_fall, lowlevel_loop])
    
    not_ready_sg = bt_nodes.NotReadyForSG()
    get_sg = bt_nodes.gen_sg(manager_policy)
    sg_fall.add_children([not_ready_sg, get_sg])
    
    move_to = bt_nodes.move_to(controller_policy, calculate_controller_reward, ctrl_rew_scale)
    safe_seq = py_trees.composites.Sequence(name='SafeSeq', memory=False)
    lowlevel_loop.add_children([safe_seq, move_to])
    
    not_safe = bt_nodes.NotSafe()
    safe_move_to = bt_nodes.safe_move_to(controller_policy, safelayer, calculate_controller_reward, ctrl_rew_scale)
    safe_seq.add_children([not_safe, safe_move_to])
    
    return root
    
    
    

def build_gpbt(manager_policy, 
             controller_policy,
             safelayer,
             controller_replay_buffer, 
             calculate_controller_reward, 
             ctrl_rew_scale):
    
    root = py_trees.composites.Selector(name='AirTreeFall', memory=False)
    safe_seq = py_trees.composites.Sequence(name='SafeSeq', memory=False)
    landmark_seq = py_trees.composites.Sequence(name='LdSeq', memory=True)
    move_seq = py_trees.composites.Sequence(name='MoveSeq', memory=True)
    compute_potential_ld = bt_nodes.ComputePotentialLds(manager_policy, controller_policy, controller_replay_buffer, 
                                                        step_size=2, obstacle_info=obstacle_info)
    root.add_children([safe_seq, landmark_seq, move_seq, compute_potential_ld])
    
    not_safe = bt_nodes.NotSafe()
    safe_move_to = bt_nodes.safe_move_to(controller_policy, safelayer, calculate_controller_reward, ctrl_rew_scale)
    safe_seq.add_children([not_safe, safe_move_to])
    
    close_to_ld = bt_nodes.CloseToLd()
    get_next_ld = bt_nodes.GetNextLd(manager_policy)
    reach_goal = bt_nodes.ReachGoal()
    landmark_seq.add_children([close_to_ld, get_next_ld, reach_goal])
    
    sg_fall = py_trees.composites.Selector(name='SGFall', memory=False)
    move_to = bt_nodes.move_to(controller_policy, calculate_controller_reward, ctrl_rew_scale)
    move_seq.add_children([sg_fall, move_to])
    
    not_ready_sg = bt_nodes.NotReadyForSG()
    get_sg = bt_nodes.gen_sg(manager_policy)
    sg_fall.add_children([not_ready_sg, get_sg])
    
    return root
    

def evaluate_policy(root,
                    blackboard,
                    eval_episodes=5,
                    ):
    blackboard.env.evaluate = True
    file_name = "bt_eval_gp"
    output_data = {"goal_success": [], "step_count": [], "collisions": []}

    with torch.no_grad():
        blackboard.avg_reward = 0.
        blackboard.avg_controller_rew = 0.
        blackboard.global_steps = 0
        blackboard.goals_achieved = 0

        for eval_ep in range(eval_episodes):
            print(eval_ep)
            obs, _ = blackboard.env.reset()
            # blackboard.env.desired_goal = np.array([0, 0])
            # blackboard.env.cur_goal = np.array([0, 0])
            blackboard.goal = blackboard.env.desired_goal
            blackboard.achieved_goal = obs["achieved_goal"]
            blackboard.state = obs["observation"]
            
            blackboard.done = False
            blackboard.success = False
            blackboard.reward = -1000
            blackboard.step_count = 0
            blackboard.collision_count = 0
            blackboard.env_goals_achieved = 0
            blackboard.sg_move_count = 0
            
            blackboard.built_landmark_graph = False
            blackboard.landmark = blackboard.achieved_goal
            blackboard.ld_idx = None
            
            blackboard.goal_list = np.array([[8, -1], [10.5, 7], [14, 16]])
            blackboard.goal_idx = 0
            tick = 0
            while not blackboard.done:
                root.tick_once()
                # print('Tick', tick)
                # print("\n")
                # print(py_trees.display.unicode_tree(root=root, show_status=True))
                # tick += 1
                # time.sleep(1.0)
            print(blackboard.success)
            output_data['collisions'].append(blackboard.collision_count)
            output_data['step_count'].append(blackboard.step_count)
            output_data['goal_success'].append(int(blackboard.success))

        avg_reward = blackboard.avg_reward / eval_episodes
        avg_controller_rew = blackboard.avg_controller_rew / blackboard.global_steps
        avg_step_count = blackboard.global_steps / eval_episodes
        avg_env_finish = blackboard.goals_achieved / eval_episodes

        print("---------------------------------------")
        print("Evaluation over {} episodes:\nAvg Ctrl Reward: {:.3f}".format(eval_episodes, avg_controller_rew))
        print("Collisions:\n Avg: {}\n Med: {}\n Max: {}\n Min: {} \n Std Dev: {}".format(
            np.average(output_data['collisions']), 
            np.median(output_data['collisions']),
            np.max(output_data['collisions']),
            np.min(output_data['collisions']), 
            np.std(output_data['collisions'])))
        print("Goals achieved: {:.1f}%".format(100*avg_env_finish))
        print("Steps to finish:\n Avg: {}\n Med: {}\n Max: {}\n Min: {} \n Std Dev: {}".format(
            np.average(output_data['step_count']), 
            np.median(output_data['step_count']), 
            np.max(output_data['step_count']),
            np.min(output_data['step_count']), 
            np.std(output_data['step_count'])))
        print("---------------------------------------")

        blackboard.env.evaluate = False

        output_df = pd.DataFrame(output_data)
        output_df.to_csv(os.path.join("./navigation/bt_comparison/"+ file_name + ".csv"), float_format="%.4f", index=False)

        return avg_reward, avg_controller_rew, avg_step_count, avg_env_finish


def run(args):
    dt = 0.1 
    vehicle_name = "Drone1"
    client = airsim.MultirotorClient()
    client.confirmConnection()
    if args.type_of_env == "small":
        airobjects.destroy_objects(client)
        airobjects.spawn_walls(client, -200, 200, -17)
        airobjects.spawn_obstacles(client, -17)
    else:
        build_blocks_world(client=client, load=True)
    
    env = AirWrapperEnv(gym.make(args.env_name, client=client, dt=dt, vehicle_name=vehicle_name, type_of_env=args.type_of_env))

    max_action = float(env.action_space.high[0])

    train_ctrl_policy_noise = args.train_ctrl_policy_noise
    train_ctrl_noise_clip = args.train_ctrl_noise_clip

    train_man_policy_noise = args.train_man_policy_noise
    train_man_noise_clip = args.train_man_noise_clip

    
    high = np.array((10., 10.))
    low = - high
    

    man_scale = (high - low) / 2
    absolute_goal_scale = 0

    if args.absolute_goal:
        no_xy = False
    else:
        no_xy = True
    obs, _ = env.reset()

    goal = obs["desired_goal"]
    state = obs["observation"]

    controller_goal_dim = obs["achieved_goal"].shape[0]

    torch.cuda.set_device(args.gid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_name = "{}_{}_{}".format(args.env_name, args.algo, args.seed)
    output_data = {"frames": [], "reward": [], "dist": []}

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    state_dim = state.shape[0]
    goal_dim = goal.shape[0]
    action_dim = env.action_space.shape[0]

    
    calculate_controller_reward = utils.get_reward_function(env, args.env_name,
                                                            absolute_goal=args.absolute_goal,
                                                            binary_reward=args.binary_int_reward)

    safe_layer = SafetyLayer(env, device='cuda', load_ckpt_dir=args.load_safety_dir)

    controller_buffer = utils.ReplayBuffer(maxsize=args.ctrl_buffer_size,
                                           reward_func=calculate_controller_reward,
                                           reward_scale=args.ctrl_rew_scale)
    manager_buffer = utils.ReplayBuffer(maxsize=args.man_buffer_size)

    controller_policy = higl.Controller(
        state_dim=state_dim,
        goal_dim=controller_goal_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_lr=args.ctrl_act_lr,
        critic_lr=args.ctrl_crit_lr,
        no_xy=no_xy,
        absolute_goal=args.absolute_goal,
        policy_noise=train_ctrl_policy_noise,
        noise_clip=train_ctrl_noise_clip,
    )

    manager_policy = higl.Manager(
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=controller_goal_dim,
        actor_lr=args.man_act_lr,
        critic_lr=args.man_crit_lr,
        candidate_goals=args.candidate_goals,
        correction=not args.no_correction,
        scale=man_scale,
        goal_loss_coeff=args.goal_loss_coeff,
        absolute_goal=args.absolute_goal,
        absolute_goal_scale=absolute_goal_scale,
        landmark_loss_coeff=args.landmark_loss_coeff,
        delta=args.delta,
        policy_noise=train_man_policy_noise,
        noise_clip=train_man_noise_clip,
        no_pseudo_landmark=args.no_pseudo_landmark,
        automatic_delta_pseudo=args.automatic_delta_pseudo,
        planner_start_step=args.planner_start_step,
        planner_cov_sampling=args.landmark_sampling,
        planner_clip_v=args.clip_v,
        n_landmark_cov=args.n_landmark_coverage,
        planner_initial_sample=args.initial_sample,
        planner_goal_thr=args.goal_thr,
    )

    if args.load_replay_buffer is not None:
        manager_buffer.load(args.load_replay_buffer + "_manager.npz")
        controller_buffer.load(args.load_replay_buffer + "_controller.npz")
        print("Replay buffers loaded")

    if args.load:
        try:
            manager_policy.load(args.load_dir, args.env_name, args.load_algo, args.version, args.seed)
            controller_policy.load(args.load_dir, args.env_name, args.load_algo, args.version, args.seed)
            print("Loaded successfully.")
        except Exception as e:
            print(e, "Loading failed.")

    # Logging Parameters
    evaluations = []

    # Novelty PQ and novelty algorithm
    if args.algo == 'higl' and args.use_novelty_landmark:
        if args.novelty_algo == 'rnd':
            novelty_pq = utils.PriorityQueue(args.n_landmark_novelty,
                                             close_thr=args.close_thr,
                                             discard_by_anet=args.discard_by_anet)
            if args.load_replay_buffer is not None:
                elems = np.load(args.load_replay_buffer + 'novelty_pq.npy', allow_pickle=True)
                novelty_pq.elems = elems.tolist()
                novelty_pq.update_tensors()
            rnd_input_dim = state_dim if not args.use_ag_as_input else controller_goal_dim
            RND = higl.RandomNetworkDistillation(rnd_input_dim, args.rnd_output_dim, args.rnd_lr, args.use_ag_as_input)
            print("Novelty PQ is generated")
        else:
            raise NotImplementedError
    else:
        novelty_pq = None
        RND = None
    blackboard = build_blackboard(env, args.manager_propose_freq)
    bt_func_dict = {"baseline":build_baseline_bt,
                    "expert":build_expert_bt,
                    "goal_change_expert":build_goal_change_bt,
                    "gp":build_gpbt}
    func_build_bt = bt_func_dict[args.bt_type]
    
    root = func_build_bt(manager_policy, controller_policy, safe_layer,
                         controller_buffer, calculate_controller_reward, 1.0)

    # py_trees.display.render_dot_tree(root, name='baseline_bt', target_directory='./figs/')
    # sys.exit(0)
    
    # Final evaluation
    avg_ep_rew, avg_controller_rew, avg_steps, avg_env_finish, = \
        evaluate_policy(root, blackboard)
