import csv
import engine
import mcts
import random
from constants import WHITE, BLACK, EMPTY, NONE, DRAW
from numba import jit

def index_stack_pop(index_stack, n_agents):
    if len(index_stack) == 0:
        index_stack = list(range(n_agents))
        random.shuffle(index_stack)
    index = index_stack.pop()
    return (index, index_stack)

def tournament(agents, names, n):
    n_agents = len(agents)
    scores = [0] * n_agents
    games = [0] * n_agents
    index_stack = list(range(n_agents))
    random.shuffle(index_stack)
    for i in range(n):
        print("Game {}/{}".format(i+1, n))
        (index1, index_stack) = index_stack_pop(index_stack, n_agents)
        (index2, index_stack) = index_stack_pop(index_stack, n_agents)
        f_agent1 = agents[index1]
        f_agent2 = agents[index2]
        winner = play_game(f_agent1, f_agent2)
        games[index1] += 1
        games[index2] += 1
        if winner == WHITE:
            scores[index1] += 1
        elif winner == BLACK:
            scores[index2] += 1
    for i in range(n_agents):
        scores[i] = scores[i] / max(games[i], 1)
    l = list(zip(names, games, scores))
    l = sorted(l, key=lambda x: -x[1])
    for (name, games, score) in l:
        print("{}: {}, {:.4f}".format(name, games, score))

def play_game(f_agent1, f_agent2):
    (board, player, winner) = engine.new_game()
    move = -1
    while not engine.is_game_over(winner):
        if player[0] == WHITE:
            move = f_agent1(board, player, winner, move)
        else:
            move = f_agent2(board, player, winner, move)
        engine.do_move(board, player, winner, move)
    return winner[0]

def get_mcts_agent(n, C, L):
    def f(board, player, winner, move):
        return mcts.get_mcts_move(board, player, winner, n, C, L)[0]
    return f

if __name__ == "__main__":
    agents = []
    names = []
    n = 100000
    for i in range(6):
        for j in range(6):
            C = i / 5
            L = j / 5
            names.append("MCTS: C: {:.4f} L: {:.4f}".format(C, L))
            agents.append(get_mcts_agent(n, C, L))
    tournament(agents, names, 250)
