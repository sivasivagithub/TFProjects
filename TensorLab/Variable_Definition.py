import tensorflow as tf

graph5 = tf.Graph()

with graph5.as_default() :

    v = tf.Variable(0)
    print("graph")
    update = tf.compat.v1.assign(v, v + 1)
    init_op = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session(graph=graph5) as sess:

    sess.run(init_op)

    print(sess.run(v))

    for _ in range(3):
        sess.run(update)
        print(sess.run(v))