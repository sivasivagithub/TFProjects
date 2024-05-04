import tensorflow as tf

print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist

(tr_images, tr_labels), (te_images, te_labels) = mnist.load_data()

tr_images = tr_images.reshape(60000,28,28,1)
tr_images = tr_images/255.0

te_images = te_images.reshape(10000,28,28,1)
te_images = te_images/255.0

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu', input_shape= (28,28,1)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(tr_images,tr_labels, epochs=5)

test_loss, test_acc  = model.evaluate(te_images, te_labels)

print ("text acc")

print(test_acc)

print(te_labels[:100])