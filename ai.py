from constants import WHITE, BLACK, EMPTY, NONE, DRAW
from mcts import get_mcts_move
from minimax import get_minimax_move
from numba import jit
import engine
import math
import numpy as np
import random

def get_best_move(board: np.ndarray, player: np.ndarray, winner: np.ndarray, n):
    (move, white_score, depth) = get_mcts_move(board, player, winner, n, math.sqrt(2), 0.5)
    return (move, white_score, depth)