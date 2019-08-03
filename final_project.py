import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from random import randint
from os import environ

environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.reset_default_graph()
image_height = 28
image_width = 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()


class CNN:
    def __init__(self, image_height, image_width, channels, num_classes):
        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[
                                          None, image_height, image_width, channels], name="inputs")
        conv_layer_1 = tf.layers.conv2d(self.input_layer, filters=32, kernel_size=[
                                        2, 2], padding="same", activation=tf.nn.relu)
        pooling_layer_1 = tf.layers.max_pooling2d(
            conv_layer_1, pool_size=[2, 2], strides=2)
        conv_layer_2 = tf.layers.conv2d(pooling_layer_1, filters=32, kernel_size=[
                                        2, 2], padding="same", activation=tf.nn.relu)
        pooling_layer_2 = tf.layers.max_pooling2d(
            conv_layer_2, pool_size=[2, 2], padding="same", strides=2)
        flattening_pooling = tf.layers.flatten(pooling_layer_2)
        dense_layer = tf.layers.dense(
            flattening_pooling, 1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(dense_layer, rate=0.4, training=True)
        outputs = tf.layers.dense(dropout, num_classes)
        self.choice = tf.argmax(outputs, axis=1)
        self.probability = tf.nn.softmax(outputs)
        self.labels = tf.placeholder(dtype=tf.float32, name="labels")
        self.accuracy, self.accuracy_op = tf.metrics.accuracy(
            self.labels, self.choice)
        one_hot_labels = tf.one_hot(indices=tf.cast(
            self.labels, dtype=tf.int32), depth=num_classes)
        self.loss = tf.losses.softmax_cross_entropy(
            onehot_labels=one_hot_labels, logits=outputs)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
        self.train_operation = optimizer.minimize(
            loss=self.loss, global_step=tf.train.get_global_step())


steps = 20000
batch_size = 16

# added
test_img = x_test[randint(0, 20)]
plt.imshow(test_img)
plt.show()
test_img = test_img.reshape(-1, 28, 28, 1)
# added

x_train = x_train.reshape(-1, image_height, image_width, 1)
cnn = CNN(28, 28, 1, 10)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    step = 0
    while step < steps:  # this is where the while loop
        sess.run((cnn.train_operation, cnn.accuracy_op), feed_dict={
                 cnn.input_layer: x_train[step:step+batch_size], cnn.labels: y_train[step:step+batch_size]})
        step += batch_size
    # added
    print(sess.run(cnn.choice, feed_dict={cnn.input_layer: test_img}))
    # added
