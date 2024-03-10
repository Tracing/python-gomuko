import engine
import numpy as np
import tensorflow as tf

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

class PolicyNetwork(tf.keras.Model):
    def __init__(self):
        super(Regressor, self).__init__()
        self.encoder = Encoder()
        self.dense1 = tf.keras.layers.Dense(80, activation="relu")
        self.dense2 = tf.keras.layers.Dense(255, activation="softmax")
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
