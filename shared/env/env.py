import gymnasium as gym
import numpy as np
import shared.map.airmap_objects as airobjects
import shared.map.blocks_tree_generator as blocks_gen
import time
import airsim
import math
from shared.higl.utils import sens_rel_to_global

class AirWrapperEnv():
    def __init__(self, base_env):
        self.evaluate = False
        self.base_env = base_env
        self.obs_info = self.base_env.unwrapped.obs_info
        self.goal_dim = self.base_env.unwrapped.goal_dim
        self.reset_count = 0
        self.sens_idx = 4
        self.num_sensors = 8
        self.goal_list = np.array([[8, 8], [8, -8], [-8, -8], [-8, 8],
                                   [8, 0], [-8, 0], [0, 8], [0, -8]])
    
    def set_init_pos(self, pos):
        self.base_env.set_init_pos(pos * 10)
        
    def reset(self):
        self.count = 0
        self.locked = False
        self.prev_max = 0
        if self.evaluate:
            self.desired_goal = np.array([6.5, 8])
            # self.reset_count = 3
            # self.desired_goal = self.goal_list[self.reset_count]
        else:
            valid_goal = False
            while not valid_goal:
                self.desired_goal = np.random.uniform((-10, -10), (10, 10))
                test_goal = (self.desired_goal * 10).tolist()
                for obstacle in self.obs_info:
                    valid_goal = not airobjects.inside_object(test_goal, obstacle)
                    if not valid_goal:
                        break
        self.prev_goal = self.base_env.unwrapped.init_pos / 10
        self.cur_goal = self.base_env.unwrapped.init_pos / 10
        obs, info = self.base_env.reset()
        obs[:self.goal_dim] /= 10
        self.reset_count += 1
        self.reset_count %= 8
        
        next_obs = {
            'observation': obs.copy(),
            'achieved_goal': obs[:self.goal_dim],
            'desired_goal': self.desired_goal,
        }
        info['is_success'] = False
        return next_obs, info
    
    def step(self, action):
        self.count += 1
        obs, rew, done, trunc, info = self.base_env.step(action)
        obs[:self.goal_dim] /= 10
        obs[:self.goal_dim] -= self.prev_goal
        next_obs = {
            'observation': obs.copy(),
            'achieved_goal': obs[:self.goal_dim],
            'desired_goal': self.desired_goal,
        }
        rew += self._get_reward(obs) 
        info['is_success'] = rew > -0.5
        
        return next_obs, rew, done or self.count >= 500, trunc, info
    
    def change_goal(self, new_goal):
        self.prev_goal = self.cur_goal
        self.cur_goal = new_goal
        self.desired_goal = self.cur_goal - self.prev_goal
    
    def seed(self, sd):
        self.base_env.unwrapped.seed(sd)
    
    def _get_reward(self, obs):
        return -np.linalg.norm(obs[:self.goal_dim] - self.desired_goal)
    
    def get_num_constraints(self):
        return 8

    def get_constraint_values(self, state):
        # print(self.locked)
        temp_thresh = np.clip(1 - np.linalg.norm(state[2:4]) / 10, 0, 0.9)
        norm = np.linalg.norm(state[2:4])
        # self.locked = False
        if not self.locked:
            self.thresh = temp_thresh
            constr_val = np.array(state[self.sens_idx:self.sens_idx+self.num_sensors] - self.thresh)
            if np.any(constr_val > 0):
                self.prev_max = np.max(constr_val)
                self.locked = True
        else:
            # temp_const = np.array(state[4:12] - temp_thresh)
            constr_val = np.array(state[self.sens_idx:self.sens_idx+self.num_sensors] - self.thresh)
            cur_max = np.max(constr_val)
            if cur_max < self.prev_max:
                self.locked = False
            if np.max(state[self.sens_idx:self.sens_idx+8]) < 0.9:
                self.prev_max = cur_max
        
        
        return constr_val
    
    @property
    def action_space(self):
        return self.base_env.action_space
    
    @property
    def observation_space(self):
        return self.base_env.observation_space
        

class AirSimEnv(gym.Env):
    def __init__(self, client, dt, vehicle_name="Drone1", randomize_start=False, type_of_env="training") -> None:
        self.client = client
        self.dt = dt
        self.vehicle_name = vehicle_name
        self.randomize_start = randomize_start
        self.obs_info = blocks_gen.obstacle_info if type_of_env == "transfer" else airobjects.obstacle_info
        self.init_pos = np.array([0, 0])
        
        
        if type_of_env == "training":
            self.z = -30
        else:
            self.z = -15
        
        self.goal_dim = 2
        self._sensor_range = 20
        
        self.observation_space = gym.spaces.Box(-200, 200, shape=(13,), dtype=float)
        self.action_space = gym.spaces.Box(-10, 10, shape=(2,), dtype=float)
    
    def set_init_pos(self, pos):
        self.init_pos = pos
    
    def reset(self, seed=None, options=None):
        # self.client.simPause(False)
        self.client.reset()
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        
        # Fixed Z Altitude
        self.client.moveToZAsync(self.z, velocity=20, vehicle_name=self.vehicle_name).join()
        if np.any(self.init_pos != 0):
            self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(float(self.init_pos[0]), float(self.init_pos[1]), self.z), airsim.to_quaternion(0, 0, 0)), True) 
        
        if self.randomize_start:
            valid_goal = False
            while not valid_goal:
                pos = np.random.uniform((-90, -90), (90, 90))
                test_goal = (pos).tolist()
                for obstacle in self.obs_info:
                    valid_goal = not airobjects.inside_object(test_goal, obstacle, buf=5)
                    if not valid_goal:
                        break
            self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(pos[0], pos[1], self.z), airsim.to_quaternion(0, 0, 0)), True) 
        # self.client.simPause(True)
        
        self.obs, z_val = self._get_obs()
        
        return self.obs, {"z_val":z_val}
    
    def step(self, action):
        # self.client.moveByVelocityAsync(float(action[0]), 
        #                             float(action[1]), 
        #                             0, 
        #                             duration=self.dt).join()
        # self.client.simPause(False)
        # self.client.moveByVelocityZAsync(float(action[0]), 
        #                             float(action[1]), 
        #                             self.z,
        #                             yaw_mode={'is_rate':False, 'yaw_or_rate':0},
        #                             duration=self.dt).join()
        self.client.moveByVelocityZAsync(float(action[0]), 
                                    float(action[1]), 
                                    self.z,
                                    drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(is_rate=False),
                                    duration=self.dt).join()
        # self.client.simPause(True)
        # self.client.simContinueForTime(self.dt)
        # norm = np.linalg.norm(action)
        # if norm > 0:
        #     unit_vec = action / norm
        # else:
        #     unit_vec = action
        # waypt_x = self.obs[0] + 50*unit_vec[0]
        # waypt_y = self.obs[1] + 50*unit_vec[1]
        # self.client.moveToPositionAsync(float(waypt_x), float(waypt_y), self.z, velocity=float(norm), timeout_sec=0.1,
        #                             drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(is_rate=False)).join()
        # self.client.simContinueForTime(self.dt)
        # time.sleep(0.1)
        self.obs, z_val = self._get_obs()
        
        return self.obs, 0, False, False, {"z_val":z_val}
    
    def _get_obs(self):
        state = self.client.simGetGroundTruthKinematics(self.vehicle_name)
        pos = state.position
        vel = state.linear_velocity
        ori = state.orientation
        q = np.array([ori.w_val, ori.x_val, ori.y_val, ori.z_val])
        yaw = math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]*q[2] + q[3]*q[3]))
        readings = self._sensor_readings()
        # print('Original\n', readings[:-1])
        rd_conv = sens_rel_to_global(readings, -yaw)
        readings[:-1] = rd_conv

        pos_np = np.array([pos.x_val, pos.y_val, vel.x_val, vel.y_val])
        return np.concatenate([pos_np, readings.flat]), pos.z_val
    
    def _sensor_readings(self):
        sens_reads = np.zeros(9)
        for i in range(8):
            sensor_name = "Distance"+str(i)
            dst = self.client.getDistanceSensorData(distance_sensor_name=sensor_name, vehicle_name=self.vehicle_name).distance
            if dst <= self._sensor_range:
                sens_reads[i] = (self._sensor_range - dst) / self._sensor_range
        sens_reads[-1] = float(self.client.simGetCollisionInfo(self.vehicle_name).has_collided)
        # sens_reads[-1] = np.any(sens_reads[:8] > 0.97)
        return sens_reads
    
    def seed(self, sd):
        np.random.seed(sd)
