import numba
import numpy as np
import math
import mcts
import random
import time
import heapq
import engine
import tensorflow as tf
from constants import WHITE, BLACK, EMPTY, NONE, DRAW
from numba import jit

class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv11 = tf.keras.layers.Conv2D(40, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", kernel_regularizer='l2')
        self.conv12 = tf.keras.layers.Conv2D(40, kernel_size=(3, 3), padding="same", activation="relu", kernel_regularizer='l2')
        self.conv13 = tf.keras.layers.Conv2D(40, kernel_size=(3, 3), padding="same", activation="relu", kernel_regularizer='l2')
        
        self.conv21 = tf.keras.layers.Conv2D(80, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", kernel_regularizer='l2')
        self.conv22 = tf.keras.layers.Conv2D(80, kernel_size=(3, 3), padding="same", activation="relu", kernel_regularizer='l2')
        self.conv23 = tf.keras.layers.Conv2D(80, kernel_size=(3, 3), padding="same", activation="relu", kernel_regularizer='l2')
        
        self.conv31 = tf.keras.layers.Conv2D(120, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", kernel_regularizer='l2')
        self.conv32 = tf.keras.layers.Conv2D(120, kernel_size=(3, 3), padding="same", activation="relu", kernel_regularizer='l2')
        self.conv33 = tf.keras.layers.Conv2D(120, kernel_size=(3, 3), padding="same", activation="relu", kernel_regularizer='l2')

        self.batchnorm11 = tf.keras.layers.BatchNormalization()
        self.batchnorm12 = tf.keras.layers.BatchNormalization()
        self.batchnorm13 = tf.keras.layers.BatchNormalization()
        self.batchnorm21 = tf.keras.layers.BatchNormalization()
        self.batchnorm22 = tf.keras.layers.BatchNormalization()
        self.batchnorm23 = tf.keras.layers.BatchNormalization()
        self.batchnorm31 = tf.keras.layers.BatchNormalization()
        self.batchnorm32 = tf.keras.layers.BatchNormalization()
        self.batchnorm33 = tf.keras.layers.BatchNormalization()

    def call(self, x):
        x = self.conv11(x)
        x = self.batchnorm11(x)
        x2 = self.conv12(x)
        x2 = self.batchnorm12(x2)
        x2 = self.conv13(x2)
        x2 = self.batchnorm13(x2)
        x = tf.add(x, x2)
        x = self.conv21(x)
        x = self.batchnorm21(x)
        x2 = self.conv22(x)
        x2 = self.batchnorm22(x2)
        x2 = self.conv23(x2)
        x2 = self.batchnorm23(x2)
        x = tf.add(x, x2)
        x = self.conv31(x)
        x = self.batchnorm31(x)
        x2 = self.conv32(x)
        x2 = self.batchnorm32(x2)
        x2 = self.conv33(x2)
        x2 = self.batchnorm33(x2)
        x = tf.add(x, x2)
        return x

class Regressor(tf.keras.Model):
    def __init__(self):
        super(Regressor, self).__init__()
        self.encoder = Encoder()
        self.dense1 = tf.keras.layers.Dense(80, activation="relu")
        self.dense2 = tf.keras.layers.Dense(1, activation="sigmoid")
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dropout2 = tf.keras.layers.Dropout(0.5)

    def call(self, x):
        x = self.encoder(x)
        x = tf.reshape(x, (-1, 4 * 120))
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return x

class HeuristicFunction:
    def __init__(self, weights_path="weights.h5"):
        self.regressor = Regressor()
        self.regressor(tf.ones((1, 15, 15, 2)))
        self.regressor.load_weights(weights_path)

    def _board_to_tensor(self, board):
        board_white = board == WHITE
        board_black = board == BLACK
        board_transformed = np.zeros((1, 2, 15, 15), dtype=np.int8)
        board_transformed[0, 0] = board_white
        board_transformed[0, 1] = board_black
        board_transformed = np.reshape(board_transformed, (1, 15, 15, 2))
        return tf.convert_to_tensor(board_transformed, tf.float32)
    
    def call(self, board, for_white):
        xs = self._board_to_tensor(board)
        ys = self.regressor.call(xs)
        if not for_white:
            ys = 1 - ys
        return ys[0][0]