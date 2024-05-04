import tensorflow as tf


graph6 = tf.Graph()

with graph6.as_default():

    a = tf.compat.v1.placeholder(tf.int32)

    b = a+1

with tf.compat.v1.Session(graph=graph6) as sess:

    result = sess.run(b, feed_dict={a:2})

    print(result)
