import csv
import numba
import numpy as np
import math
import random
import time
import heapq
from engine import *
from constants import WHITE, BLACK, EMPTY, NONE, DRAW
from numba import jit

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def monte_carlo_evaluation(board: np.ndarray, player: np.ndarray, winner: np.ndarray, n: int):
    score = 0
    for _ in range(n):
        _board = duplicate(board)
        _player = duplicate(player)
        _winner = duplicate(winner)
        do_random_moves(_board, _player, _winner, 225)
        if _winner[0] == player[0]:
            score += 1
        elif _winner[0] == DRAW:
            score += 0.5
    return score / n

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def mcts_evaluation(board: np.ndarray, player: np.ndarray, winner: np.ndarray, n: int, C: float=math.sqrt(2), weights=None):
    results = mcts(board, player, winner, n, C, weights)
    (tree, rewards, children, n_nodes, largest_depth) = results
    return rewards[0][0] / n

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def do_random_moves_and_record(board: np.ndarray, player: np.ndarray, winner: np.ndarray, n: int):
    moves = get_valid_moves(board)
    done_moves_white = []
    done_moves_black = []
    random.shuffle(moves)
    spaces_left = np.sum(board == EMPTY)
    for i in range(n):
        if player[0] == WHITE:
            done_moves_white.append(moves[i])
        else:
            done_moves_black.append(moves[i])
        do_move(board, player, winner, moves[i], check_for_draw=spaces_left == 1)
        spaces_left -= 1
        if winner[0] != NONE:
            break
    return (done_moves_white, done_moves_black)

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def is_winning_intersection(board: np.ndarray, player: np.ndarray, winner: np.ndarray, row: int, column: int):
    return has_target_connection_length(board, player[0], row, column, 4)

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def get_rule_based_move(board: np.ndarray, player: np.ndarray, winner: np.ndarray):
    moves = get_valid_moves(board)
    best_score = -1
    best_move = 0
    for move in moves:
        (row, column) = (move // 15, move % 15)
        score = (1 if is_winning_intersection(board, player, winner, row, column) else 0) + random.random() * 0.5
        if score > best_score:
            best_score = score
            best_move = move
    return best_move

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def do_rule_based_moves_and_record(board: np.ndarray, player: np.ndarray, winner: np.ndarray, n: int):
    done_moves_white = []
    done_moves_black = []
    while not is_game_over(winner) and n > 0:
        move = get_rule_based_move(board, winner, player)
        if player[0] == WHITE:
            done_moves_white.append(move)
        else:
            done_moves_black.append(move)
        do_move(board, player, winner, move)
        n -= 1
    return (done_moves_white, done_moves_black)    

VISITS = 0
ID = 1
PARENT_ID = 2
MOVE = 3

AMAF_WHITE_SCORE = 0
AMAF_BLACK_SCORE = 1
AMAF_VISITS = 2

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def mcts_selection(board, player, winner, tree, children, rewards, moves, C, L, amaf_table):
    current = 0
    depth = 0
    while len(children[current]) > 0 and len(moves[current]) == 0 and winner[0] == NONE:
        best_score = -1
        best_node = 0
        for i in children[current]:
            reward = rewards[i][0] if player[0] == WHITE else rewards[i][1]
            amaf_reward = amaf_table[tree[i][MOVE]][AMAF_WHITE_SCORE if player[0] == WHITE else AMAF_BLACK_SCORE]
            ni = tree[i][VISITS]
            amaf_ni = max(amaf_table[tree[i][MOVE]][AMAF_VISITS], 1)
            if ni == 0:
                best_node = i
                break
            else:
                vi = (reward / ni) * (1 - L) + L * (amaf_reward / amaf_ni)
                np = tree[current][VISITS]
                score = vi + C * math.sqrt(math.log(np) / ni)
                if score > best_score:
                    best_score = score
                    best_node = i
        current = best_node
        do_move(board, player, winner, tree[current][MOVE])
        depth += 1
    return (current, depth)

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def distance_to_nearest_stone_under_3(board, move):
    (row, column) = (move // 15, move % 15)
    for d in [-1, 0, 1]:
        for d2 in [-1, 0, 1]:
            (row2, column2) = (row + d, column + d2)
            if in_board(row2, column2) and board[row2][column2] != EMPTY:
                return True
    return False

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def get_mcts_moves(board, colour):
    all_moves = get_valid_moves(board)
    return all_moves

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def mcts_expansion(board, player, winner, rewards, current, tree, children, moves, n_nodes):
    if winner[0] == NONE:
        move = moves[current].pop()
        do_move(board, player, winner, move)
        tree.append([0, n_nodes, current, move])
        rewards.append([0.0, 0.0])
        children.append([5])
        children[-1].pop()
        _moves = get_mcts_moves(board, player[0])
        _moves = list(_moves)
        moves.append(_moves)
        children[current].append(n_nodes)
        n_nodes += 1
        return (n_nodes - 1, n_nodes, 1)
    else:
        return (current, n_nodes, 0)

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def mcts_simulation(board, player, winner, current, tree, amaf_table, rollout_depth):
    if winner[0] != NONE:
        score = [1.0, 0.0] if WHITE == winner[0] else [0.5, 0.5] if winner[0] == DRAW else [0.0, 1.0]
    else:
        (white_moves, black_moves) = do_random_moves_and_record(board, player, winner, rollout_depth)
        score = [1.0, 0.0] if WHITE == winner[0] else [0.5, 0.5] if winner[0] == DRAW else [0.0, 1.0]
        for move in white_moves:
            amaf_table[move][AMAF_WHITE_SCORE] += score[0]
            amaf_table[move][AMAF_VISITS] += 1
        for move in black_moves:
            amaf_table[move][AMAF_BLACK_SCORE] += score[1]
            amaf_table[move][AMAF_VISITS] += 1
    return score

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def mcts_backpropogation(tree, rewards, current, score):
    finished = False
    while not finished:
        finished = current == 0
        rewards[current][0] += score[0]
        rewards[current][1] += score[1]
        tree[current][VISITS] += 1
        current = tree[current][PARENT_ID]

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def mcts(board: np.ndarray, player: np.ndarray, winner: np.ndarray, n: int, C: float, L: float, last_move=-1):
    rollout_depth = 225
    tree = [[0, 0, -1, last_move]]
    rewards = [[0.0, 0.0]]
    children = [[5]]
    children[0].pop()
    _moves = get_mcts_moves(board, player[0])
    random.shuffle(_moves)
    moves = [list(_moves)]
    n_nodes = 1
    largest_depth = 0
    i = 0
    amaf_table = []
    (root_board, root_player, root_winner) = duplicate_game(board, player, winner)

    for i in range(225):
        amaf_table.append([0.0, 0.0, 0.0])
    while i < n:
        (board, player, winner) = duplicate_game(root_board, root_player, root_winner)
        #Selection
        (current, depth) = mcts_selection(board, player, winner, tree, children, rewards, moves, C, L, amaf_table)
        #Expansion
        (current, n_nodes, delta_depth) = mcts_expansion(board, player, winner, rewards, current, tree, children, moves, n_nodes)
        #Simulation
        score = mcts_simulation(board, player, winner, current, tree, amaf_table, rollout_depth)
        #Backpropogation
        mcts_backpropogation(tree, rewards, current, score)
        #Housekeeping
        depth += delta_depth
        largest_depth = max(largest_depth, depth)
        i += 1
    return (tree, rewards, children, n_nodes, largest_depth, amaf_table)

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def get_mcts_move(board: np.ndarray, player: np.ndarray, winner: np.ndarray, n: int, C: float, L: float, last_move=-1):
    (tree, rewards, children, n_nodes, depth, amaf_table) = mcts(board, player, winner, n, C, L, last_move)
    most_visits = -1
    most_visits_move = 0
    for i in range(len(children[0])):
        child = children[0][i]
        visits = (1 - L) * tree[child][VISITS] + L * amaf_table[tree[child][MOVE]][AMAF_VISITS]
        if visits > most_visits:
            most_visits_move = tree[child][MOVE]
            most_visits = visits
    return (most_visits_move, rewards[0][0] / n, depth)
