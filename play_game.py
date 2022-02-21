from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import py_2048

def show_board(game):
    fig = plt.figure(figsize=(4,4))
    subplot = fig.add_subplot(1,1,1)

    board = game.board
    score = game.score
    result = np.zeros([4,4])
    for x in range(4):
        for y in range(4):
            result[y][x] = board[y][x]

    sns.heatmap(result, square=True, cbar=False, annot=True, linewidth=2, xticklabels=False, yticklabels=False, vmax=512, vmin=0, fmt='.5g', cmap='prism_r', ax=subplot).set_title('2048 game!')
    plt.show()

    print('score: {0:.0f}'.format(score))

def human_play():
    game = py_2048.Game()
    show_board(game)
    while True:
        select_action = int(input())
        if(py_2048.is_invalid_action(game.board.tolist(),select_action)):
            print('cannot action!')
            continue
        r = game.action(select_action)
        clear_output(wait=True)
        show_board(game)

human_play()