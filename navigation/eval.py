import torch

import numpy as np

import higl.utils as utils
import higl.higl as higl

import airsim

import gymnasium as gym
from env import *

from env.env import AirWrapperEnv

import airmap.airmap_objects as airobjects



def evaluate_policy(env,
                    env_name,
                    manager_policy,
                    controller_policy,
                    calculate_controller_reward,
                    ctrl_rew_scale,
                    controller_replay_buffer,
                    novelty_pq,
                    manager_propose_frequency=10,
                    eval_idx=0,
                    eval_episodes=5,
                    ):
    print("Starting evaluation number {}...".format(eval_idx))
    env.evaluate = True
    prefix = ""
    file_name = prefix + "_evaluation_0_8"
    output_data = {"goal_success": [], "step_count": []}

    with torch.no_grad():
        avg_reward = 0.
        avg_controller_rew = 0.
        global_steps = 0
        goals_achieved = 0
        collision_list = []
        step_list = []

        for eval_ep in range(eval_episodes):
            obs, _ = env.reset()

            goal = obs["desired_goal"]
            achieved_goal = obs["achieved_goal"]
            state = obs["observation"]
            done = False
            step_count = 0
            env_goals_achieved = 0
            while not done:
                if step_count % manager_propose_frequency == 0:
                    subgoal = manager_policy.sample_goal(state, goal)


                step_count += 1
                global_steps += 1
                action = controller_policy.select_action(state, subgoal)
                new_obs, reward, done, trunc, info = env.step(action)
                
                is_success = info['is_success']
                if is_success:
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
            output_data['step_count'].append(step_count)
            output_data['goal_success'].append(env_goals_achieved)

        avg_reward /= eval_episodes
        avg_controller_rew /= global_steps
        avg_step_count = global_steps / eval_episodes
        avg_env_finish = goals_achieved / eval_episodes

        print("---------------------------------------")
        print("Evaluation over {} episodes:\nAvg Ctrl Reward: {:.3f}".format(eval_episodes, avg_controller_rew))
        if "Gather" in env_name:
            print("Avg reward: {:.1f}".format(avg_reward))
        else:
            print("Goals achieved: {:.1f}%".format(100*avg_env_finish))
        print("Steps to finish:\n Avg: {}\n Med: {}\n Max: {}\n Min: {} \n Std Dev: {}".format(np.average(output_data['step_count']), np.median(output_data['step_count']), np.max(output_data['step_count']),
                                                                                np.min(output_data['step_count']), np.std(output_data['step_count'])))
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
        # output_df.to_csv(os.path.join("./approach_safe/results_"+ file_name + ".csv"), float_format="%.4f", index=False)

        return avg_reward, avg_controller_rew, avg_step_count, avg_env_finish, \
               final_x, final_y, final_z, \
               final_subgoal_x, final_subgoal_y, final_subgoal_z


def run(args):
    dt = 0.1 
    vehicle_name = "Drone1"
    client = airsim.MultirotorClient()
    client.confirmConnection()
    airobjects.spawn_walls(client, -200, 200, -32)
    airobjects.spawn_obstacles(client, -32)
    
    env = AirWrapperEnv(gym.make(args.env_name, client=client, dt=dt, vehicle_name=vehicle_name))

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


    # Final evaluation
    avg_ep_rew, avg_controller_rew, avg_steps, avg_env_finish, \
    final_x, final_y, final_z, final_subgoal_x, final_subgoal_y, final_subgoal_z = \
        evaluate_policy(env, args.env_name, manager_policy, controller_policy, calculate_controller_reward,
                        args.ctrl_rew_scale, controller_buffer, novelty_pq, args.manager_propose_freq, len(evaluations))
