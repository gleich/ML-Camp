import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.reset_default_graph()
test_constant = tf.constant(10, dtype=tf.int32)
add_one_operation = test_constant + 1
tf.Session()

with tf.Session() as sess:
    print(sess.run(add_one_operation))
