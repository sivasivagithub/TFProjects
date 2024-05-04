import tensorflow as tf

graph1 = tf.Graph() # creating Graph

with graph1.as_default():

  a = tf.constant([2],name='constant_a') #Defining constant or Tensor
  b = tf.constant([8],name='constant_b')


  sess = tf.compat.v1.Session(graph=graph1) #Defining session

  result = sess.run(a+b) #Printing result

  print("a + b = c = ")
  print(result)

  sess.close()