import itertools
import math
import random

import gym
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from gym import spaces
import tensorflow as tf
#from utils import helpers as utl

# from torch_encoder import FeatureExtractor
# import torch
import copy


class GridNavi(gym.Env):
    def __init__(self, num_cells=5, num_steps=15):
        super(GridNavi, self).__init__()

        self.seed()
        self.num_cells = num_cells
        self.num_states = num_cells ** 2

        self._max_episode_steps = num_steps
        self.step_count = 0

        self.observation_space = spaces.Box(low=0, high=self.num_cells - 1, shape=(10,)) # was 2
        self.action_space = spaces.Discrete(5)  # noop, up, right, down, left
        self.task_dim = 2
        self.belief_dim = 25

        # possible starting states
        self.starting_state = [0.0, 0.0]

        # goals can be anywhere except on possible starting states and immediately around it
      #  self.possible_goals = list(itertools.product(range(num_cells), repeat=2))
      #  self.possible_goals.remove((0, 0))
      #  self.possible_goals.remove((0, 1))
      #  self.possible_goals.remove((1, 1))
      #  self.possible_goals.remove((1, 0))

       # self.possible_goals = [(3,2),(4,2),(2,3),(2,4),(num_cells-1,num_cells-1),(num_cells-1,num_cells-2),(num_cells-2,num_cells-2),(num_cells-2,num_cells-1)]
        self.possible_goals = [(num_cells-1,num_cells-1),(num_cells-1,num_cells-2),(num_cells-2,num_cells-2),(num_cells-2,num_cells-1)]
     #   self.possible_goals = [(0,2),(4,2)]
    #    self.task_dim = 2
        self.num_tasks = self.num_states

        # reset the environment state
        self._env_state = np.array(self.starting_state)
        # reset the goal
        self._goal = self.reset_task()
     #   self.nn = FeatureExtractor(2,10)
        # reset the belief
       # self._belief_state = self._reset_belief()

    def reset_task(self, task=None):
        if task is None:
            self._goal = np.array(random.choice(self.possible_goals))
        else:
            self._goal = np.array(task)
      #  self._reset_belief()
        return self._goal

    def get_task(self):
        return self._goal.copy()


    def reset(self):
        obs_array =[]
        self.step_count = 0
        self._env_state = np.array(self.starting_state)
#         state = torch.from_numpy(self._env_state)
#         out=self.nn.forward(state.float())
#         out = out.detach().numpy()
        return np.array(self._env_state)

    def state_transition(self, action):
        """
        Moving the agent between states
        """

        if action == 1:  # up
            self._env_state[1] = min([self._env_state[1] + 1, self.num_cells - 1])
        elif action == 2:  # right
            self._env_state[0] = min([self._env_state[0] + 1, self.num_cells - 1])
        elif action == 3:  # down
            self._env_state[1] = max([self._env_state[1] - 1, 0])
        elif action == 4:  # left
            self._env_state[0] = max([self._env_state[0] - 1, 0])

        return self._env_state

    def step(self, action):

        if isinstance(action, np.ndarray) and action.ndim == 1:
            action = action[0]
        assert self.action_space.contains(action)

        done = False

        # perform state transition
        state = self.state_transition(action)

        # check if maximum step limit is reached
        self.step_count += 1
        if self.step_count >= self._max_episode_steps:
            done = True

        # compute reward
        if self._env_state[0] == self._goal[0] and self._env_state[1] == self._goal[1]:
            reward = 1.0
            
       #elif self._env_state[0] == 0 and self._env_state[1] in [1,2,3,4]:
       #    reward = -0.5
            
       #elif self._env_state[1] == 4 and self._env_state[0] in [0,1,2]:
       #    reward = -0.5
        
        else:
            reward = -0.2

        # update ground-truth belief
        #self.update_belief(self._env_state, action)

        task = self.get_task()
       # task_id = self.task_to_id(task)
        info = {'task': task,
                'task_id': 111,
                'belief': 11}
        
#         encoded = self.nn.encoder(state)
#         state = tf.concat([encoded[0], encoded[1]], 0)
#         state = np.asarray(state).astype('float32')
 #       state = torch.from_numpy(self._env_state)
        #out=self.nn.forward(state.float())
 #       out=self.nn.forward(state.float())
 #       out = out.detach().numpy()
    
        return state, reward, done, info



  