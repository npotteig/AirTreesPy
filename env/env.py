import gymnasium as gym
import numpy as np

class AirWrapperEnv():
    def __init__(self, base_env):
        self.evaluate = False
        self.base_env = base_env
        self.goal_dim = self.base_env.unwrapped.goal_dim
    
    def reset(self):
        self.count = 0
        if self.evaluate:
            self.desired_goal = np.array([6.5, 8])
        else:
            self.desired_goal = np.random.uniform((-10, -10), (10, 10))
            
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
        next_obs = {
            'observation': obs.copy(),
            'achieved_goal': obs[:self.goal_dim],
            'desired_goal': self.desired_goal,
        }
        rew += self._get_reward(obs) 
        info['is_success'] = rew > -2.5
        
        return next_obs, rew, done or self.count >= 500, trunc, info
    
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
        
        self.observation_space = gym.spaces.Box(-200, 200, shape=(4,), dtype=float)
        self.action_space = gym.spaces.Box(-5, 5, shape=(2,), dtype=float)
    
    def reset(self, seed=None, options=None):
        self.client.reset()
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        
        # Fixed Z Altitude
        self.client.moveToZAsync(-30, velocity=15, vehicle_name=self.vehicle_name).join()
        
        obs = self._get_obs()
        
        return obs, {}
    
    def step(self, action):
        # self.client.moveByVelocityAsync(float(action[0]), 
        #                             float(action[1]), 
        #                             0, 
        #                             duration=self.dt).join()
        self.client.moveByVelocityZAsync(float(action[0]), 
                                    float(action[1]), 
                                    -30, 
                                    duration=self.dt).join()
        obs = self._get_obs()
        
        return obs, 0, False, False, {}
    
    def _get_obs(self):
        state = self.client.simGetGroundTruthKinematics(self.vehicle_name)
        pos = state.position
        vel = state.linear_velocity
        pos_np = np.array([pos.x_val, pos.y_val, vel.x_val, vel.y_val])
        return pos_np
    
    def seed(self, sd):
        np.random.seed(sd)