from ai import get_best_move
from constants import WHITE, BLACK, EMPTY, NONE, DRAW
import engine
import numpy as np
import tensorflow as tf

class Gomoku_MDP:
    def __init__(self, is_white, n, C, L):
        self.n = n
        self.C = C
        self.L = L
        self.reset(is_white)
    
    def reset(self, is_white):
        (self.board, self.player, self.winner) = engine.new_game()
        self.is_white = is_white
        self.last_reward = 0

        if not self.is_white:
            (ai_move, _, _) = get_best_move(self.board, self.player, self.winner, self.n, self.C, self.L)
            engine.do_move(self.board, self.player, self.winner, ai_move)

    def get_state(self):
        _board = np.copy(self.board)
        if not self.is_white:
            _board[self.board == WHITE] = BLACK
            _board[self.board == BLACK] = WHITE
        return (tf.convert_to_tensor(_board.reshape((1, 15, 15, 1)), dtype=tf.float32), self.winner[0])

    def get_reward(self):
        return self.last_reward

    def get_actions(self):
        return engine.get_valid_moves(self.board)

    def do_action(self, action: int):
        if not action in self.get_actions() or self.has_finished():
            self.last_reward = 0
        else:
            engine.do_move(self.board, self.player, self.winner, action)
            if self.winner[0] == NONE:
                (ai_move, _, _) = get_best_move(self.board, self.player, self.winner, self.n, self.C, self.L)
                engine.do_move(self.board, self.player, self.winner, ai_move)
                if self.winner[0] == NONE or self.winner[0] == DRAW:
                    self.last_reward = 0
                else:
                    self.last_reward = -1
            elif self.winner[0] == DRAW:
                self.last_reward = 0
            else:
                self.last_reward = 1

        return (self.get_state(), self.last_reward)

    def has_finished(self):
        return self.winner[0] != NONE