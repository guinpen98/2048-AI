import numpy as np

class Game:
    def __init__(self):
        self.start()

    def start(self):
        self.board = np.zeros([4,4]) # 盤面の初期化
        self.score = 0 # スコアの初期化
        x = np.random.randint(0,4) 
        y = np.random.randint(0,4)
        # 盤面にランダムに2を生成
        self.board[y][x] = 2 
        while(True):
            x = np.random.randint(0,4) 
            y = np.random.randint(0,4)
            if self.board[y][x]==0:
                self.board[y][x] = 2
                break
    
    def action(self,action):
        reward = 0 # 報酬の初期化

        if action==1: # 左にスワイプ
            for y in range(4):
                z_cnt = 0 # 値が0であるマスの数
                prev = -1 # 0を抜かした左隣のマスの値
                for x in range(4):
                    if self.board[y][x]==0: # 値が0の時
                        z_cnt += 1
                    elif self.board[y][x]==prev: # 左隣のマスと値が同じ時
                        z_cnt += 1
                        self.board[y][x-z_cnt] *= 2 # 左隣にあるマスの値を二倍にする
                        self.board[y][x] = 0
                        reward += self.board[y][x-z_cnt]
                        self.score += self.board[y][x-z_cnt]
                        prev = -1
                    else:
                        prev = self.board[y][x]
                        if z_cnt==0: # 値が0であるマスの分、値を移動させる
                            continue
                        self.board[y][x] = 0
                        self.board[y][x-z_cnt] = prev
                        
        elif action==2: # 右にスワイプ
            for y in range(4):
                z_cnt = 0 # 値が0であるマスの数
                prev = -1 # 0を抜かした右隣のマスの値
                for x in range(4):
                    if self.board[y][3-x]==0: # 値が0の時
                        z_cnt += 1
                    elif self.board[y][3-x]==prev: # 右隣のマスと値が同じ時
                        z_cnt += 1
                        self.board[y][3-x+z_cnt] *= 2 # 右隣にあるマスの値を二倍にする
                        self.board[y][3-x] = 0
                        reward += self.board[y][3-x+z_cnt]
                        self.score += self.board[y][3-x+z_cnt]
                        prev = -1
                    else:
                        prev = self.board[y][3-x]
                        if z_cnt==0: # 値が0であるマスの分、値を移動させる
                            continue
                        self.board[y][3-x] = 0
                        self.board[y][3-x+z_cnt] = prev

        elif action==3: # 上にスワイプ
            for x in range(4):
                z_cnt = 0 # 値が0であるマスの数
                prev = -1 # 0を抜かした上隣のマスの値
                for y in range(4):
                    if self.board[y][x]==0: # 値が0の時
                        z_cnt += 1
                    elif self.board[y][x]==prev: # 上隣のマスと値が同じ時
                        z_cnt += 1
                        self.board[y-z_cnt][x] *= 2 # 上隣にあるマスの値を二倍にする
                        self.board[y][x] = 0
                        reward += self.board[y-z_cnt][x]
                        self.score += self.board[y-z_cnt][x]
                        prev = -1
                    else:
                        prev = self.board[y][x]
                        if z_cnt==0: # 値が0であるマスの分、値を移動させる
                            continue
                        self.board[y][x] = 0
                        self.board[y-z_cnt][x] = prev
                        
        elif action==4: # 下にスワイプ
            for x in range(4):
                z_cnt = 0 # 値が0であるマスの数
                prev = -1 # 0を抜かした下隣のマスの値
                for y in range(4):
                    if self.board[3-y][x]==0: # 値が0の時
                        z_cnt += 1
                    elif self.board[3-y][x]==prev: # 下隣のマスと値が同じ時
                        z_cnt += 1
                        self.board[3-y+z_cnt][x] *= 2 # 下隣にあるマスの値を二倍にする
                        self.board[3-y][x] = 0
                        reward += self.board[3-y+z_cnt][x]
                        self.score += self.board[3-y+z_cnt][x]
                        prev = -1
                    else:
                        prev = self.board[3-y][x]
                        if z_cnt==0: # 値が0であるマスの分、値を移動させる
                            continue
                        self.board[3-y][x] = 0
                        self.board[3-y+z_cnt][x] = prev

        while(True):
            y = np.random.randint(0,4)
            x = np.random.randint(0,4)
            if self.board[y][x]==0:
                if np.random.random() < 0.2: # 20%の確率で4が生成される
                    self.board[y][x] = 4
                else:
                    self.board[y][x] = 2
                break

        return reward

def is_invalid_action(state,action): # 選択したアクションが無効かどうか

    if action==1:
        for y in range(4):
            z_cnt = 0
            prev = -1
            for x in range(4):
                if state[y][x]==0:
                    z_cnt += 1
                elif state[y][x]==prev or z_cnt > 0:
                    return False
                else:
                    prev = state[y][x]

    if action==2:
        for y in range(4):
            z_cnt = 0
            prev = -1
            for x in range(4):
                if state[y][3-x]==0:
                    z_cnt += 1
                elif state[y][3-x]==prev or z_cnt > 0:
                    return False
                else:
                    prev = state[y][3-x]

    if action==3:
        for x in range(4):
            z_cnt = 0
            prev = -1
            for y in range(4):
                if state[y][x]==0:
                    z_cnt += 1
                elif state[y][x]==prev or z_cnt > 0:
                    return False
                else:
                    prev = state[y][x]

    if action==4:
        for x in range(4):
            z_cnt = 0
            prev = -1
            for y in range(4):
                if state[3-y][x]==0:
                    z_cnt += 1
                elif state[3-y][x]==prev or z_cnt > 0:
                    return False
                else:
                    prev = state[3-y][x]
    
    return True


