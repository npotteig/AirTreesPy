import gymnasium as gym
import numpy as np

class SimpleEnv(gym.Env):
    def __init__(self, vel, dt, eval=False, goal=None) -> None:
        self.vel = vel
        self.dt = dt
        self.goal = goal
        self.eval = eval
        
        self.observation_space = gym.spaces.Box(-500, 500, shape=(3,), dtype=float)
        self.action_space = gym.spaces.Box(-400, 400, shape=(3,), dtype=float)
    
    def reset(self, seed=None, options=None):
        self.count = 0
        self.position = np.array([0.0, 0.0, -30.0])
        
        return self._get_obs(), self._get_info()
    
    def step(self, action, skip_move=False):
        self._move(action)
        obs = self._get_obs()
        rew = self._get_reward(obs)
        self.count += 1
        if self.eval:
            terminated = rew > -2.5 or self.count >= 500
        else: 
            terminated = self.count >= 100
        return obs, rew, terminated, False, self._get_info()
    
    def _get_obs(self):
        return self.position
    
    def _get_reward(self, obs):
        return -np.linalg.norm(obs[:3] - self.goal)
    
    def _get_info(self):
        return {}
    
    def _move(self, target):
        disp = target - self.position
        dist = np.linalg.norm(disp)
        
        # Ensures that agent does not skip over target
        if dist < (self.vel * self.dt):
            self.position = target
        else:
            force_dir = (disp / dist) * (self.vel * self.dt) if dist > 0 else np.array([0, 0, 0])
            self.position += force_dir