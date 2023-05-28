import random
import numpy as np
import copy
import math

class SmartStorage:
    def __init__(self, N, M, K, env_type="static"):
        # env_type            --> Environment types static, dynamic
        # N                   --> Number of products
        # M = (n1, n2)        --> Map Layout
        # k                   --> No of Previous Orders

        self.N = N
        self.M = M
        self.env_type = env_type
        self.K = K
        self.season = 1
        self.init_env()

    def init_env(self):
        # Initialize Order Probability Vector 
        self.order_p = [(i+0.5)/(self.N+0.5) for i in range(self.N)]
        self.order_p_for_s2 = reversed(self.order_p)
        #self.order_p = [random.random() for i in range(self.N)]
        self.actions =[]

        # Possible Reposition Actions
        for i in range(1, self.N):
            for j in range(i+1,self.N+1):
                self.actions.append((i,j))

        # initial L Orders
        self.last_k_orders = [self._get_order() for i in range(self.K)]


        # Total Actions / Action Space
        self.n_actions = int(((self.N*self.N-self.N)/2)+1)

        # State Vector Size
        self.n_state = self.N+self.N
        #self.generate_all_maps()

    def reset(self):
        # Initialize Map Layout
        product_list = [i for i in range(1, self.N+1)]
        random.shuffle(product_list)
        self.map = np.array(product_list).reshape(self.M)
        self.last_k_orders = [self._get_order() for i in range(self.K)]
        p_maps = self.generate_P_maps()
        self.order = self.last_k_orders[-1]
        return self.map, self.last_k_orders, p_maps
    
    def step(self, act, p_vec):
        if(act != "None"):
            item1, item2 = self.actions[act]
            reloc_cost = self.reposition_cost(item1, item2)
            self.relocate_items(item1, item2) 
        else:
            reloc_cost = 0
            
        # Get new Order and add it into Order History
        travel_cost = self.travel_cost(p_vec)
        self.order = self._get_order()
        self.last_k_orders.append(self.order)
        self.last_k_orders.pop(0)

        p_maps = self.generate_P_maps()

        return self.map, p_maps, self.last_k_orders, 2*travel_cost+0.5*reloc_cost

    def get_state_info(self):
        # 1. Order probability
        # 2. Order
        # 3. Last K Orders
        # 3. Storage map
        return self.order_p, self.order, self.last_k_orders, self.map

    def calculate_reward(self, item1, item2):
        
        pass

    def travel_cost(self, p_vector):
        # 1. Total travel Time Cost
        cost = 0
        for i in range(self.N):
            x1, y1 = self._get_item_Loc(i+1)
            cost += p_vector[i]*(x1+y1)
        return cost

    def reposition_cost(self, item1, item2):
        # 2. Reposition Cost 
        x1, y1 = self._get_item_Loc(item1)
        x2, y2 = self._get_item_Loc(item2)

        cost = abs(x1 - x2)+abs(y1 - y2)

        return cost

    def _get_item_Loc(self, Item):
        for i in range(self.M[0]):
            for j in range(self.M[1]):
                if(Item == self.map[i,j]):
                    return (i,j)

    def _get_order(self):
        order=[1*(random.random()<self.order_p[i]) for i in range(self.N)]
        return order

    def relocate_items(self, item1, item2, update=True):
        x1, y1 = self._get_item_Loc(item1)
        x2, y2 = self._get_item_Loc(item2)

        new_map = copy.deepcopy(self.map)
        if(update):
            self.map[x1, y1] = item2
            self.map[x2, y2] = item1
        else:
            new_map[x1, y1] = item2
            new_map[x2, y2] = item1
            return new_map
    
    def generate_data_for_lstm(self, K):
        self.order_p = [random.random() for i in range(self.N)]
        orders = np.array([self._get_order() for i in range(K)])
        #orders = np.transpose(orders)

        return orders, self.order_p

    def generate_P_maps(self):
        P_maps = []
        for act in self.actions:
            item1, item2 = act
            pmap=self.relocate_items(item1, item2, update=False)
            P_maps.append(pmap)
        P_maps.append(self.map)
        return P_maps

    def generate_all_maps(self):
        s = self.N
        m=[]
        self.all_Ps={}
        all_y=self.gen(s, m)
        id=0
        for i in range(0, len(all_y), s):
            self.all_Ps[tuple(all_y[i:i+s])]=id
            id+=1

    def gen(self, s, m):
        out = []
        if(len(m)==s):
            return m
        else:
            for i in range(1, s+1):
                if(i in m):
                    continue
                else:
                    mn=copy.deepcopy(m)
                    mn.append(i)
                    d = self.gen(s, mn)
                    out = out + d
            return out 

