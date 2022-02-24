from pickletools import optimize
import torch
import torch.nn as nn
from torch import optim

import deep_q_network
import play_game

n_mid = 2048

net = deep_q_network.Net(n_mid)

loss_fnc = nn.MSELoss()
optimizer = optim.RMSprop(net.parameters(), lr=0.01)  # 最適化アルゴリズム

brain = deep_q_network.Brain(net,loss_fnc,optimizer)

