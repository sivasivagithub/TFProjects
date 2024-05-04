import tensorflow as tf
import numpy as np
from tensorflow import keras

# y = 2x - 1.

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])]) # Defining a neural network

model.compile(optimizer='sgd', loss = 'mean_squared_error')  # Compiling the neural network with loss and optimizer

xs = np.array([-1.0,1.0,2.0,3.0,4.0], dtype=float) # Provide data to the neural network
ys = np.array([-3.0,-1.0,1.0,3.0,5.0],dtype=float)

model.fit(xs, ys, epochs= 1000) #Training neural network

print(model.predict([8.0]))




