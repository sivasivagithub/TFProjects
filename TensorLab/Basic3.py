import tensorflow as tf

graph3 =  tf.Graph()

with graph3.as_default():

    Matrix_one = tf.constant([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    Matrix_two = tf.constant([[2, 2, 2], [2, 2, 2], [2, 2, 2]])

    add_1_operation = tf.add(Matrix_one, Matrix_two)
    add_2_operation = Matrix_one + Matrix_two

with tf.compat.v1.Session(graph =graph3) as sess:

    result = sess.run(add_1_operation)
    print ("Defined using tensorflow function :")
    print(result)
    result = sess.run(add_2_operation)
    print ("Defined using normal expressions :")
    print(result)