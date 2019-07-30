import tensorflow as tf
from os import environ

environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

hetjkrgkgbghjsfdbghjkfsdbgdfhjksbgdhjkfsbgjksdf

tf.reset_default_graph()
input_data = tf.placeholder(dtype=tf.int32, shape=None)
double_operation = input_data * 10

with tf.Session() as sess:
    print(sess.run(double_operation, feed_dict={input_data: [[1, 2], [3, 4]]}))
