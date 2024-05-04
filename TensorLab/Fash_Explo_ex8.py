import tensorflow as tf
import numpy as np                    # numpy is used for array operations and scientific calculation
import matplotlib.pyplot as plt         # plotting graph

print ("PRINTING VERSION")
print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
mnist1 = tf.keras.datasets.mnist # Assigning dataset to a variable

(training_images, training_labels) ,  (test_images, test_labels) = mnist1.load_data() # load data from dataset

np.set_printoptions(linewidth=150) # linewidth specifies the number of characters printed in a row

plt.imshow(training_images[0]) #plotting graph

###plt.show()
print ("PRINTING LABELS")

print(training_labels[0])

print ("PRINTING TRAINING IMAGES")

print(training_images[0])

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',  loss = 'sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print ("PRINTING CLASSIFICATION")
print(classifications[0])

print ("PRINTING TEST LABELS")
print(test_labels[0])