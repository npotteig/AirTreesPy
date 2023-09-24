import torch
import torch.optim as optim

import os
import numpy as np
import pandas as pd
from math import ceil
from collections import OrderedDict

import higl.utils as utils
import higl.higl as higl
from higl.models import ANet

import airsim

import gymnasium as gym
from env import *

from env.env import AirWrapperEnv

import airmap.airmap_objects as airobjects
from airmap.blocks_tree_generator import build_blocks_world


def run(args):
    
    if not os.path.exists("./navigation/replay_data"):
        os.makedirs("./navigation/replay_data")

    dt = 0.1 
    vehicle_name = "Drone1"
    client = airsim.MultirotorClient()
    client.confirmConnection()
    if args.type_of_env == 'small':
        airobjects.spawn_walls(client, -200, 200, -32)
        airobjects.spawn_obstacles(client, -32)
    elif args.type_of_env == 'large':
        build_blocks_world(client=client, load=True)
    
    env = AirWrapperEnv(gym.make(args.env_name, client=client, dt=dt, vehicle_name=vehicle_name), args.type_of_env)
        

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
        
    obs, info = env.reset()
    print("obs: ", obs)


    goal = obs["desired_goal"]
    state = obs["observation"]

    controller_goal_dim = obs["achieved_goal"].shape[0]


    torch.cuda.set_device(args.gid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    file_name = "{}_{}_{}".format(args.env_name, args.algo, args.seed)
    output_data = {"frames": [], 'training_collisions':[]}

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

    if args.noise_type == "ou":
        man_noise = utils.OUNoise(goal_dim, sigma=args.man_noise_sigma)
        ctrl_noise = utils.OUNoise(action_dim, sigma=args.ctrl_noise_sigma)

    elif args.noise_type == "normal":
        man_noise = utils.NormalNoise(sigma=args.man_noise_sigma)
        ctrl_noise = utils.NormalNoise(sigma=args.ctrl_noise_sigma)

    if args.load:
        try:
            manager_policy.load(args.load_dir, args.env_name, args.load_algo, args.version, args.seed)
            controller_policy.load(args.load_dir, args.env_name, args.load_algo, args.version, args.seed)
            print("Loaded successfully.")
            just_loaded = True
        except Exception as e:
            just_loaded = False
            print(e, "Loading failed.")
    else:
        just_loaded = False

    # Logging Parameters
    total_timesteps = 0
    timesteps_since_eval = 0
    timesteps_since_manager = 0
    episode_timesteps = 0
    timesteps_since_subgoal = 0
    episode_num = 0
    timesteps_since_print = 0
    done = True
    evaluations = []
    
    # Intervention Variables
    intervene = False
    actions_before_intervention = []
    intervene_index = 1
    train_collisions = 0

    ep_obs_seq = None
    ep_ac_seq = None

    # Novelty PQ and novelty algorithm
    if args.algo == 'higl' and args.use_novelty_landmark:
        if args.novelty_algo == 'rnd':
            novelty_pq = utils.PriorityQueue(args.n_landmark_novelty,
                                             close_thr=args.close_thr,
                                             discard_by_anet=args.discard_by_anet)
            rnd_input_dim = state_dim if not args.use_ag_as_input else controller_goal_dim
            RND = higl.RandomNetworkDistillation(rnd_input_dim, args.rnd_output_dim, args.rnd_lr, args.use_ag_as_input)
            print("Novelty PQ is generated")
        else:
            raise NotImplementedError
    else:
        novelty_pq = None
        RND = None
        

    while total_timesteps < args.max_timesteps:
        if done:
            # # Update Novelty Priority Queue
            # if ep_obs_seq is not None:
            #     assert ep_ac_seq is not None
            #     if args.algo == 'higl' and args.use_novelty_landmark:
            #         if args.novelty_algo == 'rnd':
            #             if args.use_ag_as_input:
            #                 novelty = RND.get_novelty(np.array(ep_ac_seq).copy())
            #             else:
            #                 novelty = RND.get_novelty(np.array(ep_obs_seq).copy())
            #             novelty_pq.add_list(ep_obs_seq, ep_ac_seq, list(novelty), a_net=a_net)
            #             novelty_pq.squeeze_by_kth(k=args.n_landmark_novelty)
            #         else:
            #             raise NotImplementedError

            if total_timesteps != 0 and not just_loaded:
                if episode_num % 10 == 0:
                    print("Episode {}, Timesteps {}".format(episode_num, total_timesteps))
                    output_data["frames"].append(total_timesteps)
                    print("Training Collisions:", train_collisions)
                    output_data['training_collisions'].append(train_collisions)
                    train_collisions = 0
                    
                # # Update RND module
                # if RND is not None:
                #     rnd_loss = RND.train(controller_buffer, episode_timesteps, args.rnd_batch_size)

                if len(manager_transition['state_seq']) != 1:
                    manager_transition['next_state'] = state
                    manager_transition['done'] = float(True)
                    manager_buffer.add(manager_transition)

            # Reset environment
            obs, info = env.reset()
            goal = obs["desired_goal"]
            achieved_goal = obs["achieved_goal"]
            state = obs["observation"]

            # ep_obs_seq = [state]  # For Novelty PQ
            # ep_ac_seq = [achieved_goal]

            done = False
            episode_reward = 0
            episode_timesteps = 0
            just_loaded = False
            episode_num += 1

            # Intervention Variables
            intervene = False
            actions_before_intervention = []
            intervene_index = 1

            subgoal = manager_policy.sample_goal(state, goal)
            timesteps_since_subgoal = 0
            manager_transition = OrderedDict({
                'state': state,
                'next_state': None,
                'achieved_goal': achieved_goal,
                'next_achieved_goal': None,
                'goal': goal,
                'action': subgoal,
                'reward': 0,
                'done': False,
                'state_seq': [state],
                'actions_seq': [],
                'achieved_goal_seq': [achieved_goal]
            })

        
        if not intervene:
            action = controller_policy.select_action(state, subgoal)
            action = ctrl_noise.perturb_action(action, -max_action, max_action)
            actions_before_intervention.append(action)
        else:
            potential = utils.calc_potential(state[4:12])
            action = np.clip(10 * potential, -10, 10) 
            
            intervene_index = 2
        action_copy = action.copy()

        next_tup, manager_reward, env_done, _, info = env.step(action_copy)
    

        # Update cumulative reward for the manager
        manager_transition['reward'] += manager_reward * args.man_rew_scale

        next_goal = next_tup["desired_goal"]
        next_achieved_goal = next_tup['achieved_goal']
        next_state = next_tup["observation"]
        collide = next_state[-1]

        # if not collide == 1:
        #     ep_obs_seq.append(next_state)
        #     ep_ac_seq.append(next_achieved_goal)

        # Append low level sequence for off policy correction
        manager_transition['actions_seq'].append(action)
        manager_transition['state_seq'].append(next_state)
        manager_transition['achieved_goal_seq'].append(next_achieved_goal)

        controller_reward = calculate_controller_reward(achieved_goal, subgoal, next_achieved_goal,
                                                        args.ctrl_rew_scale, action)
        subgoal = controller_policy.subgoal_transition(achieved_goal, subgoal, next_achieved_goal)

        if collide == 1:
            train_collisions += 1
        
        inter_temp = False
        if np.any(next_state[4:12] > 0.80):
            controller_reward += -5
            inter_temp = True

        controller_goal = subgoal
        if env_done:
            done = True

        episode_reward += controller_reward

        # Store low level transition
        if args.inner_dones:
            ctrl_done = done or timesteps_since_subgoal % args.manager_propose_freq == 0
        else:
            ctrl_done = done

        controller_transition = OrderedDict({
            'state': state,
            'next_state': next_state,
            'achieved_goal': achieved_goal,
            'next_achieved_goal': next_achieved_goal,
            'goal': controller_goal,
            'action': action,
            'reward': controller_reward,
            'done': float(ctrl_done),
            'state_seq': [],
            'actions_seq': [],
            'achieved_goal_seq': []
        })
        
        if inter_temp:
            intervene = True
        elif not inter_temp and intervene_index > 1:
            intervene = False
            actions_before_intervention = []
            intervene_index = 1
        
        controller_buffer.add(controller_transition)

        state = next_state
        goal = next_goal
        achieved_goal = next_achieved_goal

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
        timesteps_since_manager += 1
        timesteps_since_subgoal += 1
        timesteps_since_print += 1

        if timesteps_since_subgoal % args.manager_propose_freq == 0:
            manager_transition['next_state'] = state
            manager_transition['next_achieved_goal'] = achieved_goal
            manager_transition['done'] = float(done)
            manager_buffer.add(manager_transition)

            subgoal = manager_policy.sample_goal(state, goal)

            if not args.absolute_goal:
                subgoal = man_noise.perturb_action(subgoal, min_action=-man_scale, max_action=man_scale)
            else:
                subgoal = man_noise.perturb_action(subgoal, min_action=-man_scale, max_action=man_scale)

            # Reset number of timesteps since we sampled a subgoal
            timesteps_since_subgoal = 0

            # Create a high level transition
            manager_transition = OrderedDict({
                'state': state,
                'next_state': None,
                'achieved_goal': achieved_goal,
                'next_achieved_goal': None,
                'goal': goal,
                'action': subgoal,
                'reward': 0,
                'done': False,
                'state_seq': [state],
                'actions_seq': [],
                'achieved_goal_seq': [achieved_goal]
            })

    if args.save_replay_buffer is not None:
        manager_buffer.save(args.save_replay_buffer + "_manager")
        controller_buffer.save(args.save_replay_buffer + "_controller")
        output_data["frames"].append(total_timesteps)
        output_data['training_collisions'].append(train_collisions)
        
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(os.path.join("./navigation/replay_data", file_name+".csv"), float_format="%.4f", index=False)
        # np.save(args.save_replay_buffer + 'novelty_pq', novelty_pq.elems)

    print("Training finished.")
