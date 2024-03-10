from constants import WHITE, BLACK
from mcts import get_mcts_move
import engine
import math
import numpy as np
import xml.etree.ElementTree as ET

tree = ET.parse('renjunet_v10_20220611.rif')
root = tree.find('games')

games = []
results = []

column_dict = {c: ord(c) - ord('a') for c in 'abcdefghijklmno'}
row_dict = {str(i + 1): i for i in range(15)}

def to_move(t):
    row = row_dict[t[1]]
    column = column_dict[t[0]]
    return row * 15 + column

for game in root:
    result = float(game.attrib['bresult'])
    #print(result)
    for move in game:
        if move.tag == "move":
            results.append(result)
            games.append([])

            if not move is None and not move.text is None: 
                moves = move.text.split()
                for t in moves:
                    games[-1].append(to_move(t))

xs = []
ys = []

for (game, result) in zip(games, results):
    (board, player, winner) = engine.new_game()

    for move in game:
        engine.do_move(board, player, winner, move)
        board_white = board == WHITE
        board_black = board == BLACK
        board_transformed = np.zeros((1, 2, 15, 15), dtype=np.int8)
        board_transformed[0, 0] = board_white
        board_transformed[0, 1] = board_black
        board_transformed = np.reshape(board_transformed, (1, 15, 15, 2))
        xs.append(board_transformed)
        ys.append(result)

xs = np.asarray(np.concatenate(xs), dtype=np.int8)
ys = np.asarray(ys, dtype=np.float32)

print(xs.shape)
print(ys.shape)

np.save("xs.npy", xs)
np.save("ys.npy", ys)

xs = []
ys = []

i = 0
for (game, _) in zip(games, results):
    (board, player, winner) = engine.new_game()

    for move in game:
        engine.do_move(board, player, winner, move)
        board_white = board == WHITE
        board_black = board == BLACK
        board_transformed = np.zeros((1, 2, 15, 15), dtype=np.int8)
        board_transformed[0, 0] = board_white
        board_transformed[0, 1] = board_black
        board_transformed = np.reshape(board_transformed, (1, 15, 15, 2))
        xs.append(board_transformed)
        ys.append(get_mcts_move(board, player, winner, 100000, math.sqrt(2), 0.5)[1])
        i += 1
        print((i, ys[-1]))
        if i > 2500:
            break
    if i > 2500:
        break

xs = np.asarray(np.concatenate(xs), dtype=np.int8)
ys = np.asarray(ys, dtype=np.float32)

print(xs.shape)
print(ys.shape)

np.save("xs_2.npy", xs)
np.save("ys_2.npy", ys)
