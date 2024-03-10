from engine import *
from constants import WHITE, BLACK, EMPTY, NONE, DRAW
from heuristic import HeuristicFunction
from numba import jit, int64
import numpy as np

WORST = -100.0
BEST = 100.0

#@jit(locals={'player': int64[:]}, nopython=True, cache=True, boundscheck=False, fastmath=False)
def get_minimax_move(board: np.ndarray, player: np.ndarray, winner: np.ndarray, depth: int):
    moves = get_valid_moves(board)
    best_score = WORST
    best_move = 0
    is_white = player[0] == WHITE
    alpha = WORST
    beta = BEST
    heuristic_fn = HeuristicFunction()
    for move in moves:
        (_board, _player, _winner) = (np.copy(board), np.copy(player), np.copy(winner))
        do_move(_board, _player, _winner, move)
        score = minimax(_board, _player, _winner, depth-1, False, is_white, alpha, beta, heuristic_fn)
        if score > best_score:
            best_move = move
            best_score = score
    return (best_move, best_score, depth)

#@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def minimax(board: np.ndarray, player: np.ndarray, winner: np.ndarray, depth: int, maximize: bool, is_white: bool, alpha: float, beta: float, heuristic_fn):
    if winner[0] != NONE:
        white_score = 0.5 if winner[0] == DRAW else 1 if winner[0] == WHITE else 0
        return white_score if is_white else 1 - white_score
    elif depth == 0:
        return heuristic_fn.call(board, is_white)
    else:
        moves = get_valid_moves(board)
        best_score = WORST if maximize else BEST
        for move in moves:
            (_board, _player, _winner) = (np.copy(board), np.copy(player), np.copy(winner))
            do_move(_board, _player, _winner, move)
            score = minimax(_board, _player, _winner, depth-1, not maximize, is_white, alpha, beta, heuristic_fn)
            if maximize:
                best_score = max(score, best_score)
                alpha = max(alpha, best_score)
                if best_score >= beta:
                    break
            else:
                best_score = min(score, best_score)
                beta = min(beta, best_score)
                if best_score <= alpha:
                    break
        return best_score
