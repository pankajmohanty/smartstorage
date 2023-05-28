import gym
from gym import Env, spaces
import numpy as np
import math


class SmartStorageOpenAIGym(Env):
    def __init__(self, env):
        super().__init__()

        self.env = env
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.env.n_state, ))
        self.action_space = spaces.Discrete(self.env.n_actions, )

    def demand_cal(self, data):
        demand = np.mean(data, axis=0)
        return demand

    def reset(self):
        map, k_periods, p_maps = self.env.reset()
        self.dem_p = self.demand_cal(k_periods)
        state = np.concatenate((map.reshape(-1), self.dem_p))

        next_states=[]
        for m in p_maps:
            n_state= np.concatenate((m.reshape(-1), self.dem_p))
            next_states.append(n_state)
        return state, next_states

    def step(self, act):
        if(act==self.action_space.n-1):
            act="None"
        map, p_maps, k_periods, cost = self.env.step(act, self.dem_p)
        self.dem_p = self.demand_cal(k_periods)
        state = np.concatenate((map.reshape(-1), self.dem_p))
        next_states=[]
        for m in p_maps:
            n_state= np.concatenate((m.reshape(-1), self.dem_p))
            next_states.append(n_state)
        return state, -1*cost, next_states


        