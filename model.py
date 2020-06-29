import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    h_dim = 128
    def __init__(self, o_dim, a_dim, seed):
        super(Actor, self).__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(o_dim, self.h_dim * 2)
        self.fc2 = nn.Linear(self.h_dim * 2, self.h_dim)
        self.fc3 = nn.Linear(self.h_dim, a_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        h1 = F.relu(self.fc1(state))
        h2 = F.relu(self.fc2(h1))
        h3 = self.fc3(h2)
        norm = torch.norm(h3)
        return 10.0*(F.tanh(norm))*h3/norm if norm > 0 else 10*h3


class Critic(nn.Module):
    h_dim = 128
    def __init__(self, o_dim, a_dim, seed):
        super(Critic, self).__init__()
        torch.manual_seed(seed)
        self.fcs1 = nn.Linear(o_dim, self.h_dim * 2)
        self.fc2 = nn.Linear(self.h_dim * 2 + a_dim, self.h_dim * 2)
        self.fc3 = nn.Linear(self.h_dim * 2, self.h_dim)
        self.fc4 = nn.Linear(self.h_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        xs = F.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim = 1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
