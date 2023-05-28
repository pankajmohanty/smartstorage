import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMOrderPredictor(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, order_size):
        super(LSTMOrderPredictor, self).__init__()
        self.hidden_dim = hidden_dim

        self.order_embeddings = nn.Embedding(order_size, embedding_dim)

        # The LSTM takes order embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(order_size, hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.order_prob = nn.Linear(hidden_dim, order_size)

    def forward(self, x):
        #embeds = self.order_embeddings(x)
        #print(embeds.shape)
        y =  x.view(len(x), 1, -1)
        output, (h_n, c_n) = self.lstm(y)
        #print(h_n.shape)
        tag_space = self.order_prob(h_n[0, -1])
        p_order = tag_space
        #print(p_order.shape)
        return p_order

class FFNN_ACT(nn.Module):

    def __init__(self, state_size, action_size, hidden_size):
        super(FFNN_ACT, self).__init__()
        self.hidden_dim = hidden_size

        self.l1= nn.Linear(state_size, hidden_size)
        self.l2= nn.Linear(hidden_size, hidden_size)
        self.l3= nn.Linear(hidden_size, hidden_size)
        self.l4= nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x