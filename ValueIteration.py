import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

from models import FFNN_ACT
from SmartStorageEnv import SmartStorage
from gymEnv import SmartStorageOpenAIGym
import math
import random
import matplotlib.pyplot as plt



class ValueIteration:
    def __init__(self, N, M, K, n_episodes=300, eps=0.9, eps_decay=0.999):
        self.env = SmartStorageOpenAIGym(SmartStorage(N, M, K, env_type = "static"))
        self.ffnn_v = FFNN_ACT(N+N, 1, 128)

        self.opt = optim.Adam(self.ffnn_v.parameters(), lr= 0.001)
        self.criterion = nn.MSELoss()
        self.n_steps=N*2
        self.n_episodes=n_episodes
        self.eps=eps
        self.eps_decay=eps_decay
        self.N=N
        self.M=M
        self.tst=False

    def select_action(self, next_states):
        values=[]
        self.ffnn_v.eval()
        for state in next_states:
            v = self.ffnn_v(torch.from_numpy(state).float())
            values.append(v.cpu().detach().numpy()[0])
        self.eps=self.eps*self.eps_decay
        #print(values)
        #print(np.argmax(values))
        if(random.random()>self.eps):
            return np.argmax(values), np.max(values)
        else:
            ind = random.randint(0, len(next_states)-1)
            return ind, values[ind]
    def greedy_select_action(self, next_states):
        values=[]
        self.ffnn_v.eval()
        for state in next_states:
            v = self.ffnn_v(torch.from_numpy(state).float())
            values.append(v.cpu().detach().numpy())
        self.eps=self.eps*self.eps_decay
        return np.argmax(values), np.max(values)
     


    def update_model(self):
        self.ffnn_v.train()
        for s, c, v in zip(self.state_list, self.cost_list, self.next_state_value):
            self.opt.zero_grad()
            V = self.ffnn_v(torch.from_numpy(s).float())
            loss = self.criterion(V[0], torch.tensor(c+0.99*v).float())
            loss.backward()
            self.opt.step()


    def train(self):
        # Store training data
        cost_hist = []
        # Repeat for n episodes
        for i in range(self.n_episodes):
            self.state_list=[]
            self.cost_list=[]
            self.next_state_value=[]
            state, next_states = self.env.reset() 
            tCost=0
            for j in range(self.n_steps):
                # Action Selection
                a, V = self.select_action(next_states)
                # Take Step
                next_state, cost, nextn__states = self.env.step(a)
                #print(f"action: {a} Cost: {cost}, V:{V, self.env.env.map}")
                self.state_list.append(state)
                self.cost_list.append(cost)
                self.next_state_value.append(V)

                state = next_state
                next_states = nextn__states
                tCost+=cost
            self.update_model()
            cost_hist.append(tCost)
            print(f"Episode: {i}/{self.n_episodes} Cost: {tCost}")
            print(self.eps)

        return cost_hist
    def evaluate(self):
        # Repeat for n episodes
        for i in range(1):
            state, next_states = self.env.reset() 
            self.showInventoryMap()
            tCost=0
            print(f"Initial Map:{self.env.env.map}, P_Order: {self.env.env.order_p}")

            for j in range(self.n_steps):
                # Action Selection
                a, V = self.greedy_select_action(next_states)
                # Take Step
                next_state, cost, nextn__states = self.env.step(a)
                print(f"action: {a} Cost: {cost}, Map:{self.env.env.map}")

                state = next_state
                next_states = nextn__states
                tCost+=cost
            print(f"Total Cost: {tCost}")
            self.showInventoryMap()
    def showInventoryMap(self):
        IneventoryMap = np.zeros(self.M)
        plt.figure(figsize=(5,5))
        for i in range(1, self.N+1):
            x, y = self.env.env._get_item_Loc(i)
            IneventoryMap[x, y] = self.env.dem_p[i-1]
            plt.text(x, y, str(i))
        plt.colorbar()
        plt.imshow(IneventoryMap)
        plt.show()

class RollOutValueIteration(ValueIteration):
    def __init__(self, N, M, K, n_episodes=300, eps=0.9, eps_decay=0.999):
        super().__init__(N, M, K, n_episodes, eps, eps_decay)

    def train(self):
        # Store training data
        cost_hist = []
        self.roll_out_steps=3
        # Repeat for n episodes
        for i in range(self.n_episodes):
            self.state_list=[]
            self.cost_list=[]
            self.next_state_value=[]
            state, next_states = self.env.reset() 
            tCost=0
            for j in range(self.n_steps):
                state, next_states = self.env.reset() 
                a, V = self.greedy_select_action(next_states)
                _, act_cost, _ = self.env.step(a)
                # Rollout Policy
                V = 0
                for j in range(self.roll_out_steps):
                    # Take Step
                    _, cost, _ = self.env.step(len(next_states)-1)
                V += cost 
                #print(f"action: {a} Cost: {cost}, V:{V, self.env.env.map}")
                self.state_list.append(state)
                self.cost_list.append(act_cost)
                self.next_state_value.append(V)
            self.update_model()
            print(f"Episode: {i}/{self.n_episodes} Cost: {self.test()}")
            print(self.eps)
        return cost_hist
    def test(self):
        # Repeat for n episodes
        for i in range(1):
            state, next_states = self.env.reset() 
            tCost=0
            for j in range(self.n_steps):
                # Action Selection
                a, V = self.greedy_select_action(next_states)
                # Take Step
                next_state, cost, nextn__states = self.env.step(a)
                state = next_state
                next_states = nextn__states
                tCost+=cost
        return tCost
                





