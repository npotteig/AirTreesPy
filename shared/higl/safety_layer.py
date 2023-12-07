from datetime import datetime
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam

from shared.higl.models import ConstraintModel
from shared.higl.utils import SafetyMemory

from tqdm import tqdm

class SafetyLayer:
    def __init__(self, env=None, device = "cpu", 
                 load_buffer_dir=None, load_ckpt_dir=None):
        self.env = env
        self.device = torch.device(device)
        
        self.load_buffer = load_buffer_dir
        
        self.correction_scale = 1.0
        if self.env:
            self.action_low = self.env.action_space.low
            self.action_high = self.env.action_space.high
            self.goal_dim = self.env.goal_dim
            self.sens_idx = self.env.sens_idx
            self.num_sensors = self.env.num_sensors
            # # init constraint model
            state_dim = self.env.observation_space.shape[0]
            action_dim = self.env.action_space.shape[0]
            self.models = [ConstraintModel(state_dim, action_dim).to(self.device)
                        for _ in range(self.env.get_num_constraints())]
        else:
            self.action_low = np.array([-10.0, -10.0])
            self.action_high = np.array([10.0, 10.0])
            state_dim = 14
            action_dim = 2
            self.models = [ConstraintModel(state_dim, action_dim).to(self.device)
                       for _ in range(8)]
        
        if load_ckpt_dir is not None:
            for i in range(len(self.models)):
                self.models[i].load_state_dict(torch.load(f"{load_ckpt_dir}/safetymodel{i}.pth"))
            print("Safety Models Loaded Successfully")
        
        # use doubles for calculations
        for model in self.models:
            model.double()
        # Mean-Squared-Error as loss criterion
        self.loss_criterion = nn.MSELoss()
        # init memory
        # self.memory = SafetyMemory(self.memory_buffer_size)
        
        self.train_step = 0
        self.eval_step = 0

    def set_train_mode(self):
        for model in self.models:
            model.train()

    def set_eval_mode(self):
        for model in self.models:
            model.eval()

    def _sample_steps(self, episodes):
        # sample episodes and push to memory
        for ep in tqdm(range(episodes)):
            obs, _ = self.env.reset()
            state = obs["observation"]
            constraints = self.env.get_constraint_values(state)
            timesteps = 0
            done = False
            near_obstacle = False
            step_reset = np.random.randint(1, 20)
            action = self.env.action_space.sample()
            while not done:
                state[:self.goal_dim] = 0
                # state[self.sens_idx-1] = 0
                # get random action
                if near_obstacle and timesteps % step_reset == 0:
                    action = self.env.action_space.sample()
                    step_reset = np.random.randint(1, 20)
                # apply action
                next_obs, _, done, _, _ = self.env.step(action)
                next_state = next_obs["observation"]
                # get changed constraint values
                next_constraints = self.env.get_constraint_values(next_state)
                # push to memory only if obstacle detected
                if np.any(state[self.sens_idx:self.sens_idx+self.num_sensors] > 0):
                    self.memory.push(state, action, constraints, next_constraints)
                    near_obstacle = True
                state = next_state
                constraints = next_constraints
                
                timesteps += 1
        

    def _calc_loss(self, model, states, actions, constraints, next_constraints):
        
        # calculate batch-dot-product via torch.einsum
        # gi = model.forward(states)
        # predicted_constraints = constraints + \
        #     torch.einsum('ij,ij->i', gi, actions)
        # alternative: calculate batch-dot-product via torch.bmm
        gi = model.forward(states).unsqueeze(1)
        actions = actions.unsqueeze(1)
        predicted_constraints = constraints + torch.bmm(gi, actions.transpose(1,2)).squeeze(1).squeeze(1)

        # alternative loss calculation
        # loss = (next_constraints - predicted_constraints) ** 2
        # return torch.mean(loss)
        loss = self.loss_criterion(next_constraints, predicted_constraints)
        return loss

    def _update_batch(self, batch):
        states, actions, constraints, next_constraints = batch
        states = torch.DoubleTensor(states).to(self.device)
        actions = torch.DoubleTensor(actions).to(self.device)
        constraints = torch.DoubleTensor(constraints).to(self.device)
        next_constraints = torch.DoubleTensor(next_constraints).to(self.device)

        losses = []
        for i, (model, optimizer) in enumerate(zip(self.models, self.optims)):
            # calculate loss
            loss = self._calc_loss(
                model, states, actions, constraints[:, i], next_constraints[:, i])
            losses.append(loss.item())
            # zero gradients
            optimizer.zero_grad()
            # backward pass
            loss.backward()
            # update optimizer
            optimizer.step()

        return np.array(losses)

    def _evaluate_batch(self, batch):
        states, actions, constraints, next_constraints = batch
        states = torch.DoubleTensor(states).to(self.device)
        actions = torch.DoubleTensor(actions).to(self.device)
        constraints = torch.DoubleTensor(constraints).to(self.device)
        next_constraints = torch.DoubleTensor(next_constraints).to(self.device)

        losses = []
        for i, model in enumerate(self.models):
            # compute losses
            loss = self._calc_loss(
                model, states, actions, constraints[:, i], next_constraints[:, i])
            losses.append(loss.item())

        return np.array(losses)

    def train(self, batch_size, lr, sample_data_episodes, buffer_size, 
              epochs, train_per_epoch, eval_per_epoch):
        start_time = time.time()
        
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.training_steps_per_epoch = train_per_epoch
        self.evaluation_steps_per_epoch = eval_per_epoch
        self.memory_buffer_size = buffer_size
        self.sample_data_episodes = sample_data_episodes
        
        self.optims = [Adam(model.parameters(), lr=self.lr)
                       for model in self.models]
        
        self.memory = SafetyMemory(self.memory_buffer_size)

        print("==========================================================")
        print("Initializing constraint model training...")
        print("----------------------------------------------------------")
        print(f"Start time: {datetime.fromtimestamp(start_time)}")
        print("==========================================================")

        print("==========================================================")
        print(f"Sample Random Episode Data for {self.sample_data_episodes} episodes")
        if self.load_buffer is not None:
            self.memory.load(f"{self.load_buffer}/safetybuffer.pkl")
            print(f"Memory Loaded with {len(self.memory)} datapoints")
        else:
            # sample random action episodes and store them in memory
            self._sample_steps(self.sample_data_episodes)
            print(f"Replay Buffer Saved with {len(self.memory)} datapoints")
        print("Finished Episode Sampling")
        print("==========================================================")
        print(f"Training Phase for {self.epochs} epochs")
        for epoch in range(self.epochs):
            # training phase
            self.set_train_mode()
            losses = []
            for _ in range(self.training_steps_per_epoch):
                batch = self.memory.sample(self.batch_size)
                loss = self._update_batch(batch)
                losses.append(loss)
            print(
                f"Finished epoch {epoch} with average loss: {np.mean(losses, axis=0)}. Running evaluation ...")
            self.train_step += 1

            # evaluation phase
            self.set_eval_mode()
            losses = []
            for _ in range(self.evaluation_steps_per_epoch):
                batch = self.memory.sample(self.batch_size)
                loss = self._evaluate_batch(batch)
                losses.append(loss)
            print(
                f"Evaluation completed, average loss {np.mean(losses, axis=0)}")
            self.eval_step += 1
            print("----------------------------------------------------------")
            
        

        print("==========================================================")
        print(
            f"Finished training constraint model. Time spent: {(time.time() - start_time) // 1} secs")
        print("==========================================================")

    def get_safe_action(self, state, action, constraints):
        self.set_eval_mode()

        state = torch.DoubleTensor(state).to(self.device)
        action = torch.DoubleTensor(action).to(self.device)
        constraints = torch.DoubleTensor(constraints).to(self.device)

        g = [model.forward(state) for model in self.models]
        # calculate lagrange multipliers
        multipliers = torch.tensor([torch.clip((torch.dot(
            gi, action) + ci) / torch.dot(gi, gi), min=0) for gi, ci in zip(g, constraints)])
        # Calculate correction; scale correction to be more agressive
        correction = torch.max(multipliers) * g[torch.argmax(multipliers)]
        
        safe_action = action - correction * self.correction_scale
        safe_action = safe_action.data.detach().cpu().numpy()
        potential_safe_action = np.clip(safe_action, self.action_low, self.action_high)
        pot_norm = np.linalg.norm(potential_safe_action)
        if pot_norm == np.linalg.norm(np.array([10, 10])):
            potential_safe_action = np.clip(13 * safe_action / np.linalg.norm(safe_action), self.action_low, self.action_high)

        return potential_safe_action

    def save(self, save_dir, models=True, replay_buffer=True):
        if models:
            for i in range(len(self.models)):
                torch.save(self.models[i].state_dict(), f"{save_dir}/safetymodel{i}.pth")
            print("Safety Models Saved")
        
        if replay_buffer:
            self.memory.save(f"{save_dir}/safetybuffer.pkl")
            print(f"Replay Buffer Saved with {len(self.memory)} datapoints")
        
