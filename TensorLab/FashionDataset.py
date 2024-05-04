import tensorflow as tf
from os import path, getcwd, chdir

def train_mnist():

    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
