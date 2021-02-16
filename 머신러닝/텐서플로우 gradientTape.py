import tensorflow as tf
import numpy as np
x = tf.ones((2, 2))
print(x)
with tf.GradientTape() as t:
  t.watch(x)
  y = tf.reduce_sum(x)
  z = tf.multiply(y, y)
print(y)
print(z)
dz_dy = t.gradient(z, y) #z를 미분하고 y의 값을 넣음
print(dz_dy)