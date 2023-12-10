import torch

import numpy as np

import shared.higl.utils as utils
import shared.higl.higl as higl
from shared.higl.safety_layer import SafetyLayer

import airsim
import math
import os
import pandas as pd

import gymnasium as gym
from shared.env import *

import time

from shared.env.env import AirWrapperEnv
from shared.world_map import BlocksMaze, BlocksTrees


def evaluate_policy(env,
                    env_name,
                    manager_policy,
                    controller_policy,
                    safelayer,
                    calculate_controller_reward,
                    ctrl_rew_scale,
                    controller_replay_buffer,
                    wrld_map,
                    manager_propose_frequency=10,
                    eval_episodes=5,
                    ):
    env.evaluate = True
    prefix = ""
    file_name = prefix + "evaluation_ansr_fps_lds_transfer_2"
    output_data = {"goal_success": [], "step_count": [], "collisions": []}

    with torch.no_grad():
        avg_reward = 0.
        avg_controller_rew = 0.
        global_steps = 0
        goals_achieved = 0
        
        start_time = time.time()
        for eval_ep in range(eval_episodes):
            print(eval_ep)
            obs, _ = env.reset()
            # env.multi_goal = True
            # goal_changes = np.array([[8, -1], [10.5, 7], [14, 16]])
            # goal_changes_idx = 0
            # env.change_goal(goal_changes[goal_changes_idx])
            
            goal = obs["desired_goal"]
            # goal = goal_changes[goal_changes_idx]
            
            achieved_goal = obs["achieved_goal"]
            state = obs["observation"]
            manager_policy.init_planner()
            manager_policy.planner.eval_build_landmark_graph(goal, controller_policy, controller_replay_buffer, start=env.prev_goal, step_size=2, world_map=wrld_map)

            # select_goal_idx = np.random.randint(0, 101)
            # Problem graph is built off of the goal chosen
            # env.desired_goal = manager_policy.planner.landmarks_cov_nov_fg[select_goal_idx].cpu().numpy()
            # goal = env.desired_goal
            ld = achieved_goal
            ld_idx = None
            cur_ld = 0
            # print(goal)
            
            done = False
            step_count = 0
            env_goals_achieved = 0
            collision_count = 0
            
            # goal_changes = np.array([[20, 20], [30, 30], [40, 40]])
            
            while not done:
                # if goal_changes_idx < len(goal_changes) - 1 and -np.linalg.norm(goal - achieved_goal, axis=-1) > -1.0 :
                #     potential_goal = env.cur_goal + 10
                #     pot_goal_10 = (potential_goal * 10).tolist()
                #     valid_goal = False
                #     for obstacle in obstacle_info:
                #         valid_goal = not airobjects.inside_object(pot_goal_10, obstacle)
                #         if not valid_goal:
                #             break
                #     while not valid_goal:
                #         potential_goal = env.cur_goal + np.random.uniform((-10, -10), (10, 10))
                #         test_goal = (potential_goal * 10).tolist()
                #         for obstacle in obstacle_info:
                #             valid_goal = not airobjects.inside_object(test_goal, obstacle)
                #             if not valid_goal:
                #                 break
                    
                    # goal_changes_idx += 1
                    # env.change_goal(goal_changes[goal_changes_idx])
                    # ld = np.array([0, 0])
                    # cur_ld = 0
                    # ld_idx = None
                    # print(env.cur_goal)
                    
                    # manager_policy.planner.eval_build_landmark_graph(env.desired_goal, controller_policy, controller_replay_buffer, start=env.prev_goal, step_size=2, obstacle_info=obstacle_info)
                if -np.linalg.norm(ld - achieved_goal, axis=-1) > -1.0 or step_count == 0:    
                    if -np.linalg.norm(goal - achieved_goal, axis=-1) <= -1.5:
                        cur_ld += 1
                        ld, ld_idx = manager_policy.planner.get_next_landmark(state, ld, goal, ld_idx)
                    else:
                        ld = goal
                    # print(ld) 
                
                if step_count % manager_propose_frequency == 0:
                    subgoal = manager_policy.sample_goal(state, ld)


                step_count += 1
                global_steps += 1

                if np.all(state[4:] == 0):
                    disp = ld - state[:2]
                    dist = np.linalg.norm(disp)
                    unit_vec = disp / dist
                    policy_action = 15 * unit_vec
                else :
                    policy_action = controller_policy.select_action(state, subgoal)
                
                
                state_copy = state.copy()
                # print(state_copy[4:-1])
                state_copy[:2] = 0
                # state_copy[4] = 0
                constraints = env.get_constraint_values(state_copy)
                
        
                # policy_action = np.array([5, 5])

                if np.any(constraints > 0):
                    action = safelayer.get_safe_action(state_copy, policy_action, constraints)
                    if np.max(state[4:12]) > 0.9:
                        potential = utils.calc_potential(state_copy[4:12])
                        action = np.clip(5 * potential, -10, 10)
                else:
                    action = policy_action     
                
                
                # action = [0, -10]
                new_obs, reward, done, trunc, info = env.step(action)
                if new_obs['observation'][-1] == 1:
                    collision_count += 1
                
                is_success = info['is_success']
                if is_success:
                    print('Success')
                    env_goals_achieved += 1
                    goals_achieved += 1
                    done = True

                goal = new_obs["desired_goal"]
                new_achieved_goal = new_obs['achieved_goal']

                new_state = new_obs["observation"]
            
                subgoal = controller_policy.subgoal_transition(achieved_goal, subgoal, new_achieved_goal)

                avg_reward += reward
                avg_controller_rew += calculate_controller_reward(achieved_goal, subgoal, new_achieved_goal,
                                                                  ctrl_rew_scale, action)
                state = new_state
                achieved_goal = new_achieved_goal
            output_data['collisions'].append(collision_count)
            output_data['step_count'].append(step_count)
            output_data['goal_success'].append(env_goals_achieved)
            
        print(time.time() - start_time)
        avg_reward /= eval_episodes
        avg_controller_rew /= global_steps
        avg_step_count = global_steps / eval_episodes
        avg_env_finish = goals_achieved / eval_episodes

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

        env.evaluate = False

        final_x = new_obs['achieved_goal'][0]
        final_y = new_obs['achieved_goal'][1]

        final_subgoal_x = subgoal[0]
        final_subgoal_y = subgoal[1]
        try:
            final_z = new_obs['achieved_goal'][2]
            final_subgoal_z = subgoal[2]
        except IndexError:
            final_z = 0
            final_subgoal_z = 0

        # output_df = pd.DataFrame(output_data)
        # output_df.to_csv(os.path.join("./navigation/paper_data/eval_results/"+ file_name + ".csv"), float_format="%.4f", index=False)

        return avg_reward, avg_controller_rew, avg_step_count, avg_env_finish, \
               final_x, final_y, final_z, \
               final_subgoal_x, final_subgoal_y, final_subgoal_z


def run(args):
    dt = 0.1 
    vehicle_name = "Drone1"
    client = airsim.MultirotorClient()
    client.confirmConnection()
    if args.type_of_env == "training":
        wrld_map = BlocksMaze(client)
    else:
        wrld_map = BlocksTrees(client, load=True)
    wrld_map.build_world()
    
    env = AirWrapperEnv(gym.make(args.env_name, client=client, dt=dt, world_map=wrld_map, vehicle_name=vehicle_name, type_of_env=args.type_of_env), world_map=wrld_map)

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

    controller_buffer = utils.ReplayBuffer(maxsize=args.ctrl_buffer_size,
                                           reward_func=calculate_controller_reward,
                                           reward_scale=args.ctrl_rew_scale)
    manager_buffer = utils.ReplayBuffer(maxsize=args.man_buffer_size)
    
    safe_layer = SafetyLayer(env, device='cuda', load_ckpt_dir=args.load_safety_dir)

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


    # Final evaluation
    avg_ep_rew, avg_controller_rew, avg_steps, avg_env_finish, \
    final_x, final_y, final_z, final_subgoal_x, final_subgoal_y, final_subgoal_z = \
        evaluate_policy(env, args.env_name, manager_policy, controller_policy, safe_layer, calculate_controller_reward,
                        args.ctrl_rew_scale, controller_buffer, wrld_map, args.manager_propose_freq)
