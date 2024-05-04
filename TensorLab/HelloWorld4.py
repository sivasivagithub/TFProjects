import tensorflow as tf

graph4 = tf.Graph()

with graph4.as_default():
    tf.compat.v1.Session

    Matrix_one = tf.constant([[2, 3], [3, 4]])
    Matrix_two = tf.constant([[2, 3], [3, 4]])

    mul_operation = tf.matmul(Matrix_one, Matrix_two)


with tf.compat.v1.Session(graph=graph4) as sess:

 result = sess.run(mul_operation)

 print ("Defined using tensorflow function :")

print(result)
