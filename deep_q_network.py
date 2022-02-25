import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F # 活性化関数

import py_2048

class Net(nn.Module):
    def __init__(self,n_mid1,n_mid2):
        super().__init__()
        self.fc1 = nn.Linear(16,n_mid1) # 盤面16
        self.fc2 = nn.Linear(n_mid1,n_mid2)
        self.fc3 = nn.Linear(n_mid2,4) # 行動4

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
        state = np.ravel(states)
        state = torch.from_numpy(state).float()
        next_state = np.ravel(next_states)
        next_state = torch.from_numpy(next_state).float()
            
        self.net.eval()  # 評価モード
        next_q = self.net.forward(next_state)
        self.net.train()  # 訓練モード
        q = self.net.forward(state)

        t = q.clone().detach()
        if is_end:
            t[:] = reward  #  エピソード終了時の正解は、報酬のみ
        else:
            t[:] = reward + self.gamma*np.max(next_q.detach().cpu().numpy(), axis=0)
            
        loss = self.loss_fnc(q, t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # def get_action(self, states):  # 行動を取得
    #     state = np.ravel(states)
    #     state = torch.from_numpy(state).float()

    #     q = self.net.forward(state) #nnからQ値を取得
    #     action = np.argmax(q.detach().cpu().numpy(), axis=0) # Q値の高い行動を選択

    #     if np.random.rand() < self.eps or py_2048.is_invalid_action(states,action):  # ランダムな行動
    #         while(True):
    #             action = np.random.randint(self.n_action)
    #             if not py_2048.is_invalid_action(states,action):
    #                 break
            
    #     if self.eps > 0.1:  # εの下限
    #         self.eps *= self.r
    #     return action

    def get_train_action(self, game):  # 行動を取得
        if np.random.rand() < self.eps:  # ランダムな行動
            while(True):
                action = np.random.randint(self.n_action)
                if not py_2048.is_invalid_action(game.board.tolist(),action):
                    break
        else:
            q = np.empty(0)
            for a in range(4):
                board_backup = copy.deepcopy(game.board)
                score_backup = game.score

                if(py_2048.is_invalid_action(game.board.tolist(), a)):
                    q = np.append(q,-10**10)
                else:
                    r = game.action(a)
                    state_new = np.ravel(game.board.tolist())
                    state_new = torch.from_numpy(state_new).float()
                    q_tmp = self.net.forward(state_new)
                    q = np.append(q,r+np.amax(q_tmp.detach().cpu().numpy(), axis=0))
                game.board = board_backup
                game.score = score_backup

            action = np.argmax(q) # Q値の高い行動を選択

            
        if self.eps > 0.1:  # εの下限
            self.eps *= self.r
        return action

    def get_action(self, game):  # 行動を取得
        q = np.empty(0)
        for a in range(4):
            board_backup = copy.deepcopy(game.board)
            score_backup = game.score

            if(py_2048.is_invalid_action(game.board.tolist(), a)):
                q = np.append(q,-10**10)
            else:
                r = game.action(a)
                state_new = np.ravel(game.board.tolist())
                state_new = torch.from_numpy(state_new).float()
                q_tmp = self.net.forward(state_new)
                q = np.append(q,r+np.amax(q_tmp.detach().cpu().numpy(), axis=0))
            game.board = board_backup
            game.score = score_backup

        return np.argmax(q) # Q値の高い行動を選択

class Ai:
    def __init__(self,brain,game):
        self.brain = brain
        self.game = game
    
    def learning(self):
        current_states = self.game.board
        action = self.brain.get_train_action(self.game)
        reward = self.game.action(action)
        self.brain.train(current_states,self.game.board,action,reward,py_2048.is_end(self.game.board.tolist()))

    def action(self):
        action = self.brain.get_action(self.game)
        reward = self.game.action(action)
