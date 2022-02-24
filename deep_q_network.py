import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F # 活性化関数

import py_2048

class Net(nn.Module):
    def __init__(self,n_mid):
        super().__init__()
        self.fc1 = nn.Linear(16,n_mid) # 盤面16
        self.fc2 = nn.Linear(n_mid,n_mid)
        self.fc3 = nn.Linear(n_mid,4) # 行動4

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc1(x)

class Brain:
    def __init__(self,net,loss_fnc,optimizer,gamma=0.9,r=0.99,lr=0.01):
        self.n_state = 16  # 状態の数
        self.n_action = 4  # 行動の数

        self.net = net  # ニューラルネットワークのモデル
        self.loss_fnc = loss_fnc  # 誤差関数
        self.optimizer = optimizer  # 最適化アルゴリズム

        self.eps = 1.0  # ε
        self.gamma = gamma  # 割引率
        self.r = r  # εの減衰率
        self.lr = lr  # 学習係数

    def train(self,states,next_states,action,reward,is_end):
        states = torch.from_numpy(states).float()
        next_states = torch.from_numpy(next_states).float()
            
        self.net.eval()  # 評価モード
        next_q = self.net.forward(next_states)
        self.net.train()  # 訓練モード
        q = self.net.forward(states)

        t = q.clone().detach()
        if is_end:
            t[:, action] = reward  #  エピソード終了時の正解は、報酬のみ
        else:
            t[:, action] = reward + self.gamma*np.max(next_q.detach().cpu().numpy(), axis=1)[0]
            
        loss = self.loss_fnc(q, t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, states):  # 行動を取得
        states = torch.from_numpy(states).float()

        q = self.net.forward(states) #nnからQ値を取得
        action = np.argmax(q.detach().cpu().numpy(), axis=1)[0] # Q値の高い行動を選択

        if np.random.rand() < self.eps or py_2048.is_invalid_action(action):  # ランダムな行動
            while(True):
                action = np.random.randint(self.n_action)
                if not py_2048.is_invalid_action(action):
                    break
            
        if self.eps > 0.1:  # εの下限
            self.eps *= self.r
        return action