from ai import get_best_move
from constants import WHITE, BLACK, EMPTY, NONE, DRAW
import engine
import math
import mcts
import time
import tkinter as tk
import tkinter.simpledialog as tksd
import tkinter.messagebox as tkmb

class Interface:
    def __init__(self, scale=1.0):
        (board, player, winner) = engine.new_game()
        get_best_move(board, player, winner, 1)

        self.root = tk.Tk()
        self.top_frame = tk.Frame(self.root)
        self.top_frame.grid(column=0, row=0)
        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.grid(column=0, row=1)
        self.canvas = tk.Canvas(self.top_frame, width=scale, height=scale)
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        self.canvas.grid()
        self.scale = scale
        self.btn_computer_vs_computer = tk.Button(self.bottom_frame, text="Play Computer vs Computer Game", command=self.run_computer_vs_computer_game)
        self.btn_computer_vs_computer.grid()
        self.btn_computer_vs_human_white = tk.Button(self.bottom_frame, text="Play Human vs Computer Game as White", command=self.run_human_vs_computer_game_white)
        self.btn_computer_vs_human_white.grid()
        self.btn_computer_vs_human_black = tk.Button(self.bottom_frame, text="Play Human vs Computer Game as Black", command=self.run_human_vs_computer_game_black)
        self.btn_computer_vs_human_black.grid()

        self.draw_lines()
        self.doing_human_vs_human_game = False
        self.doing_human_vs_computer_game = False
        self.doing_computer_vs_computer_game = False
        self.human_white = False

        self.canvas_accepts_input = False

        self.board = None
        self.player = None
        self.winner = None
        self.turn = 1
        self.n = 1

    def on_canvas_click(self, e):
        if not self.canvas_accepts_input:
            return
        else:
            (x, y) = (e.x, e.y)
            moves = engine.get_valid_moves(self.board)
            if self.canvas_accepts_input:
                if self.doing_human_vs_computer_game:
                    move = self.get_move(x, y)
                    if move in moves:
                        player_is_white = self.player[0] == WHITE
                        engine.do_move(self.board, self.player, self.winner, move)
                        self.turn += 1 if player_is_white else 0
                        self.draw_game(self.board, self.player)

                        self.canvas_accepts_input = False
                        if not engine.is_game_over(self.winner):
                            player_is_white = self.player[0] == WHITE
                            (move, white_score, depth) = get_best_move(self.board, self.player, self.winner, self.n)
                            engine.do_move(self.board, self.player, self.winner, move)
                            self.turn += 1 if player_is_white else 0
                            self.draw_game(self.board, self.player)
                        
                        if engine.is_game_over(self.winner):
                            self.doing_human_vs_computer_game = False
                            self.canvas_accepts_input = False
                            if self.winner != DRAW:
                                tkmb.showinfo("{} wins".format(self.winner), "{} wins on turn {}".format(self.winner, self.turn))
                            else:
                                tkmb.showinfo("Draw", "Game is a draw. Game ended on turn {}".format(self.turn))
                        else:
                            self.canvas_accepts_input = True


    def get_move(self, mouse_x, mouse_y):
        best_row_column = (0, 0)
        lowest_score = float('inf')
        for row in range(15):
            for column in range(15):
                (prow, pcol) = self.get_pos(row, column)
                score = abs(prow - mouse_x) + abs(pcol - mouse_y)
                if score < lowest_score:
                    lowest_score = score
                    best_row_column = (row, column)
        
        (row, column) = best_row_column
        return row * 15 + column

    def run_human_vs_human_game(self):
        turn = 1
        self.reset_canvas()
        (board, player, winner) = engine.new_game()

    def run_human_vs_computer_game_white(self):
        self.human_white = True
        self.run_human_vs_computer_game()

    def run_human_vs_computer_game_black(self):
        self.human_white = False
        self.run_human_vs_computer_game()

    def run_human_vs_computer_game(self):
        self.n = -1
        while self.n <= 0:
            self.n = tksd.askinteger("Depth", "Number of Simulations?")

        self.turn = 1
        self.reset_canvas()
        (self.board, self.player, self.winner) = engine.new_game()
        if self.human_white:
            player_is_white = self.player[0] == WHITE
            (move, white_score, depth) = get_best_move(self.board, self.player, self.winner, self.n)
            engine.do_move(self.board, self.player, self.winner, move)
            self.turn += 1 if player_is_white else 0
            self.draw_game(self.board, self.player)
        self.doing_human_vs_computer_game = True
        self.canvas_accepts_input = True

    def run_computer_vs_computer_game(self):
        self.doing_computer_vs_computer_game = True
        self.canvas_accepts_input = False
        turn = 1
        (white_n, black_n) = (-1, -1)
        while white_n <= 255:
            white_n = tksd.askinteger("Number simulations for white", "How many white simulations per move? (Minimum 225)")
        while black_n <= 255:
            black_n = tksd.askinteger("Number simulations for black", "How many black simulations per move? (Minimum 225)")
        self.reset_canvas()
        (board, player, winner) = engine.new_game()
        start = time.time()
        move = -1
        while not engine.is_game_over(winner):
            player_is_white = player[0] == WHITE
            n = white_n if player_is_white else black_n
            (move, white_score, depth) = get_best_move(board, player, winner, n)
            print("{} turn. Win chance for white: {:.4f}. Depth: {}".format("Black" if player_is_white else "White", white_score, depth))
            engine.do_move(board, player, winner, move)
            turn += 1 if player_is_white else 0
            self.draw_game(board, player)
        winner = "White" if winner[0] == WHITE else "Black" if winner[0] == BLACK else "Draw"
        if winner != DRAW:
            tkmb.showinfo("{} wins".format(winner), "{} wins on turn {}".format(winner, turn))
        else:
            tkmb.showinfo("Draw", "Game is a draw. Game ended on turn {}".format(turn))

        time_elapsed = time.time() - start
        print("White player made {} simulations per move".format(white_n))
        print("Black player made {} simulations per move".format(black_n))
        print("Game took {:.4f} seconds".format(time_elapsed))
        print("Each move took approximately {:.4f} seconds".format(time_elapsed / (turn * 2)))
        self.doing_computer_vs_computer_game = False
        self.canvas_accepts_input = False

    def _scale(self, amount):
        return amount * self.scale

    def reset_canvas(self):
        self.canvas.delete("all")
        self.draw_lines()
        self._draw_postprocessing()

    def draw_game(self, board, player):        
        for row in range(15):
            for column in range(15):
                if board[row][column] != EMPTY:
                    stone_is_white = board[row][column] == WHITE
                    self.draw_stone(row, column, stone_is_white)
        self._draw_postprocessing()

    def get_pos(self, row, column):
        dr = row * 0.06
        dc = column * 0.06
        return (self._scale(0.05 + dr), self._scale(0.05 + dc))

    def _draw_postprocessing(self):
        self.canvas.update()

    def draw_lines(self):
        for row in range(15):
            d = row * 0.06
            self.canvas.create_line(self._scale(0.05), self._scale(0.05 + d), self._scale(0.89), self._scale(0.05 + d), width=self._scale(0.0025), fill="grey")
        for column in range(15):
            d = column * 0.06
            self.canvas.create_line(self._scale(0.05 + d), self._scale(0.05), self._scale(0.05 + d), self._scale(0.89), width=self._scale(0.0025), fill="grey")
        self._draw_postprocessing()

    def draw_stone(self, row, column, is_white, is_last_move=False):
        (x, y) = self.get_pos(row, column)
        delta = self._scale(0.03) // 2
        colour = 'white' if is_white else 'black'
        outline = 'black' if not is_last_move else 'red'
        self.canvas.create_oval(x - delta, y - delta, x + delta, y + delta, fill=colour, outline=outline)

    def run(self):
        self.root.mainloop()

interface = Interface(450)
interface.run()