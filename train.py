import torch
import torch.nn as nn
from torch import optim

import deep_q_network
import py_2048
from play_game import show_board
from IPython.display import clear_output

n_mid1 = 2048
n_mid2 = 1024

net = deep_q_network.Net(n_mid1,n_mid2)

loss_fnc = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)  # 最適化アルゴリズム

brain = deep_q_network.Brain(net,loss_fnc,optimizer)

game = py_2048.Game()

ai = deep_q_network.Ai(brain,game)

# 学習

for t in range(4000):
    ai.game.reset()
    while(not py_2048.is_end(ai.game.board)):
        ai.learning()

model_path = './model/new_model_4000_Adam.pth'
torch.save(net.state_dict(),model_path)

# show_board(ai.game)
# for i in range(1):
#     ai.game.reset()
#     while(not py_2048.is_end(ai.game.board)):
#         ai.action()
#         clear_output(wait=False)
#         show_board(ai.game)
