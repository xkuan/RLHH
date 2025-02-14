import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal
import numpy as np
import numpy.random as rd


class QNet(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )
        self.explore_rate = 0.125
        self.action_dim = action_dim

    def forward(self, state):
        return self.net(state)  # Q values for multiple actions

    def get_action(self, state):
        if rd.rand() > self.explore_rate:
            action = self.net(state).argmax(dim=1, keepdim=True)
        else:
            action = torch.randint(self.action_dim, size=(state.shape[0], 1))
        return action


class QNetDuel(nn.Module):  # Dueling DQN
    """
    Critic class for **Dueling Q-network**.
    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_state = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
        )
        self.net_adv = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, 1)
        )  # advantage function value 1
        self.net_val = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, action_dim)
        )  # Q value
        self.explore_rate = 0.125
        self.action_dim = action_dim

    def forward(self, state):
        """
        The forward function for **Dueling Q-network**.
        :param state: [tensor] the input state.
        :return: the output tensor.
        """
        s_tmp = self.net_state(state)  # encoded state
        q_val = self.net_val(s_tmp)
        q_adv = self.net_adv(s_tmp)
        return q_val - q_val.mean(dim=1, keepdim=True) + q_adv  # dueling Q value

    def get_action(self, state):
        if rd.rand() > self.explore_rate:
            s_tmp = self.net_state(state)
            q_val = self.net_val(s_tmp)
            action = q_val.argmax(dim=1, keepdim=True)
        else:
            action = torch.randint(self.action_dim, size=(state.shape[0], 1))
        return action


class QNetTwinDuel(nn.Module):  # D3QN: Dueling Double DQN
    """
    Critic class for **Dueling Double DQN**.
    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_state = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
        )
        self.net_val1 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, action_dim)
        )  # q1 value
        self.net_val2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, action_dim)
        )  # q2 value
        self.net_adv1 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, 1)
        )  # advantage function value 1
        self.net_adv2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, 1)
        )  # advantage function value 1
        self.explore_rate = 0.125
        self.action_dim = action_dim
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, state):
        """
        The forward function for **Dueling Double DQN**.
        :param state: [tensor] the input state.
        :return: the output tensor.
        """
        t_tmp = self.net_state(state)
        q_val = self.net_val1(t_tmp)
        q_adv = self.net_adv1(t_tmp)
        return q_val - q_val.mean(dim=1, keepdim=True) + q_adv  # one dueling Q value

    def get_q1_q2(self, state):
        """
        TBD
        """
        s_tmp = self.net_state(state)

        q_val1 = self.net_val1(s_tmp)
        q_adv1 = self.net_adv1(s_tmp)
        q_duel1 = q_val1 - q_val1.mean(dim=1, keepdim=True) + q_adv1

        q_val2 = self.net_val2(s_tmp)
        q_adv2 = self.net_adv2(s_tmp)
        q_duel2 = q_val2 - q_val2.mean(dim=1, keepdim=True) + q_adv2
        return q_duel1, q_duel2  # two dueling Q values

    def get_action(self, state):
        s = self.net_state(state)
        q = self.net_val1(s)
        if rd.rand() > self.explore_rate:
            action = q.argmax(dim=1, keepdim=True)
        else:
            a_prob = self.soft_max(q)
            action = torch.multinomial(a_prob, num_samples=1)
        return action
