import tensorflow as tf
from os import environ

environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.reset_default_graph()

input_data = tf.placeholder(dtype=tf.float32, shape=None)
output_data = tf.placeholder(dtype=tf.float32, shape=None)

slope = tf.Variable(0.5, dtype=tf.float32)
intercept = tf.Variable(0.1, dtype=tf.float32)

model_operation = slope * input_data + intercept

error = model_operation - output_data
print("Error:", error)
squared_error = tf.square(error)
print("Squared error:", squared_error)
loss = tf.reduce_mean(squared_error)
print("Loss", loss)

init = tf.global_variables_initializer()

x_values = [0, 1, 2, 3, 4]
y_values = [1, 3, 5, 7, 9]

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(loss, feed_dict={
    input_data: x_values, output_data: y_values}))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
