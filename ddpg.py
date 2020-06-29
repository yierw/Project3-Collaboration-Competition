import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from model import Critic, Actor


class DDPGAgent:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.SmoothL1Loss()

    def __init__(self, num_agents, id, x_dim, o_dim, a_dim, lr_actor, lr_critic, gamma, seed):
        self.id = id
        self.x_dim = x_dim
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        
        # Actor Network (w/ Target Network)
        self.actor = Actor(o_dim, a_dim, seed).to(self.device)
        self.target_actor = Actor(o_dim, a_dim, seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr = lr_actor)
        self.actor_scheduler = MultiStepLR(self.actor_optimizer, milestones = [1500, 2000], gamma = 0.1)
        
        # Critic Network (w/ Target Network)
        self.critic = Critic(x_dim, num_agents * a_dim, seed).to(self.device)
        self.target_critic = Critic(x_dim, num_agents * a_dim, seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = lr_critic)
    
    def get_action(self, state, eps = 0.):
        """
        action value ranges from -1 to 1
        --
        eps = 0. no exploration
            > 0. add exploration
        """
        if random.random() > eps:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.actor.eval()
            with torch.no_grad():
                action = self.actor(state_tensor)[0].detach().cpu().numpy()
            self.actor.train()
        else:
            action = np.random.randn(self.a_dim) 
        
        return np.clip(action, -1, 1)  

    def update(self, next_x, next_a, r, d, x, a, pred_a):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        next_x = next_x.view(batch_size, -1)
        # ---------------------------- update critic ---------------------------- #
        Q_next = self.target_critic(next_x, next_a)
        Q_targets = r + self.gamma * Q_next * (1. - d)
        Q_expected = self.critic(x, a)
        critic_loss = self.loss_fn(Q_expected, Q_targets.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # ---------------------------- update actor ---------------------------- #
        actor_loss = -self.critic(x, pred_a).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
