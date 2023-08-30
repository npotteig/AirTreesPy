import gymnasium as gym
import numpy as np

class AirSimEnv(gym.Env):
    def __init__(self, client, vel, eval=False, vehicle_name="Drone1", goal=None) -> None:
        self.client = client
        self.vel = vel
        self.vehicle_name = vehicle_name
        self.goal = goal
        self.eval = eval
        
        self.observation_space = gym.spaces.Box(-500, 500, shape=(3,), dtype=float)
        self.action_space = gym.spaces.Box(-400, 400, shape=(3,), dtype=float)
    
    def reset(self, seed=None, options=None):
        self.count = 0
        self.client.reset()
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        
        # Fixed Z Altitude
        self.client.moveToZAsync(-30, velocity=self.vel).join()
        
        return self._get_obs(), self._get_info()
    
    def step(self, action, skip_move=False):
        self.client.moveToPositionAsync(float(action[0]), 
                                    float(action[1]), 
                                    float(action[2]), 
                                    velocity=self.vel)
        obs = self._get_obs()
        rew = self._get_reward(obs)
        self.count += 1
        if self.eval:
            terminated = rew > -2.5 or self.count >= 500
        else: 
            terminated = self.count >= 100
        return obs, rew, terminated, False, self._get_info()
    
    def _get_obs(self):
        pos = self.client.simGetGroundTruthEnvironment(self.vehicle_name).position
        pos_np = np.array([pos.x_val, pos.y_val, pos.z_val])
        return pos_np
    
    def _get_reward(self, obs):
        return -np.linalg.norm(obs[:3] - self.goal)
    
    def _get_info(self):
        return {}