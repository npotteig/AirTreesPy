import gymnasium as gym
import numpy as np
import airmap.airmap_objects as airobjects
import airmap.blocks_tree_generator as blocks_gen
import time

class AirWrapperEnv():
    def __init__(self, base_env):
        self.evaluate = False
        self.base_env = base_env
        self.goal_dim = self.base_env.unwrapped.goal_dim
        self.obs_info = blocks_gen.obstacle_info
    
    def reset(self):
        self.count = 0
        if self.evaluate:
            self.desired_goal = np.array([8, 1])
        else:
            valid_goal = False
            while not valid_goal:
                self.desired_goal = np.random.uniform((-10, -10), (10, 10))
                for obstacle in self.obs_info:
                    valid_goal = not airobjects.inside_object(self.desired_goal, obstacle)
        self.prev_goal = np.array([0., 0.])
        obs, info = self.base_env.reset()
        obs[:self.goal_dim] /= 10
        
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
        self.prev_goal = self.desired_goal
        self.desired_goal = new_goal - self.desired_goal
    
    def seed(self, sd):
        self.base_env.unwrapped.seed(sd)
    
    def _get_reward(self, obs):
        return -np.linalg.norm(obs[:self.goal_dim] - self.desired_goal)
    
    @property
    def action_space(self):
        return self.base_env.action_space
    
    @property
    def observation_space(self):
        return self.base_env.observation_space
        

class AirSimEnv(gym.Env):
    def __init__(self, client, dt, vehicle_name="Drone1") -> None:
        self.client = client
        self.dt = dt
        self.vehicle_name = vehicle_name
        
        self.goal_dim = 2
        self._sensor_range = 20
        
        self.observation_space = gym.spaces.Box(-200, 200, shape=(13,), dtype=float)
        self.action_space = gym.spaces.Box(-10, 10, shape=(2,), dtype=float)
    
    def reset(self, seed=None, options=None):
        self.client.reset()
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        
        # Fixed Z Altitude
        self.client.moveToZAsync(-15, velocity=15, vehicle_name=self.vehicle_name).join()
        
        obs = self._get_obs()
        
        return obs, {}
    
    def step(self, action):
        # self.client.moveByVelocityAsync(float(action[0]), 
        #                             float(action[1]), 
        #                             0, 
        #                             duration=self.dt).join()
        self.client.moveByVelocityZAsync(float(action[0]), 
                                    float(action[1]), 
                                    -15, 
                                    duration=self.dt).join()
        obs = self._get_obs()
        
        return obs, 0, False, False, {}
    
    def _get_obs(self):
        state = self.client.simGetGroundTruthKinematics(self.vehicle_name)
        pos = state.position
        vel = state.linear_velocity
        readings = self._sensor_readings()
        
        pos_np = np.array([pos.x_val, pos.y_val, vel.x_val, vel.y_val])
        return np.concatenate([pos_np, readings.flat])
    
    def _sensor_readings(self):
        sens_reads = np.zeros(9)
        for i in range(8):
            sensor_name = "Distance"+str(i)
            dst = self.client.getDistanceSensorData(distance_sensor_name=sensor_name, vehicle_name=self.vehicle_name).distance
            if dst <= self._sensor_range:
                sens_reads[i] = (self._sensor_range - dst) / self._sensor_range
        sens_reads[-1] = float(self.client.simGetCollisionInfo(self.vehicle_name).has_collided)
        return sens_reads
    
    def seed(self, sd):
        np.random.seed(sd)