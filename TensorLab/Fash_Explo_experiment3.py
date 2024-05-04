import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist # Assigning dataset to a variable
print ("STARTING LOAD MNIST DATA")

(training_images, training_labels), (test_images, test_labels) = mnist.load_data() # load data from dataset

np.set_printoptions(linewidth=400)

print ("STARTING PLOTTING TRAINING IMAGES")

plt.imshow(training_images[0])


print ("PRINTING LABELS")
for j in range(15):
    print("Training value of j  " + str(j) +" label   "+str(training_labels[j] ))
    print(training_labels[j])
    print(training_images[j])
    print("Test value of j  " + str(j) + " label   " + str(training_labels[j]))
    print(test_labels[j])
    print(test_images[j])

'''
print ("PRINTING IMAGES")
print(training_images[0])


training_images  = training_images / 255.0
test_images = test_images / 255.0

"""
for j in range(15):
    print("value of j  " + str(j) +" label   "+str(training_labels[j] ))
    print(test_labels[j])
    print(test_images[j])
"""

print ("STARTING MODEL DEFINITION")
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

for j in range(15):
    print("value of j  " + str(j) +" label   "+str(training_labels[j] ))
    print(training_labels[j])
    print(training_images[j])


print ("STARTING MODEL COMPILATION WITH LOSS AND ACCURACY")
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

print ("STARTING MODEL TRAINING BY FIT")
model.fit(training_images, training_labels, epochs=5)
print ("STARTING MODEL EVALUATION")
model.evaluate(test_images, test_labels)

print ("STARTING MODEL PREDICTION")
classifications = model.predict(test_images)

print ("PRINTING CLASSIFICATION")
print(classifications[0])
print ("PRINTING TEST")
print(test_labels[0])
'''