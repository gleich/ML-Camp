import tensorflow as tf


class Convnet:

    def __init__(self, image_height, image_width, channels, num_classes):
        self.input_layer = tf.placeholder(
            dtype=float32, shape=[None, image_height, image_width, channels])
        print(self.input_layer.shape)

        conv_layer_1 = tf.layers.conv2d(self.input_layer, filters=32, kernel_size=[
                                        5, 5], padding="same", activation=tf.nn.relu)
        print(conv_layer_1.shape)

        pooling_layer_1 = tf.layers.max_pooling2d(
            conv_layer_1, pool_size=[2, 2], strides=2)
        print(pooling_layer_1.shape)
        
        conv_layer_2 = tf.layers.conv2d(pooling_layer_1, filters=3, kernel_size=[
                                        5, 5], activation=tf.nn.relu, padding="same")
        print(conv_layer_2)
        
        pooling_layer_2 = tf.layers.max_pooling2d(
            conv_layer_2, pool_size=[2, 2], strides=2)
        print(pooling_layer_2.shape)
        
        flattened_pooling = tf.layers.flatten(pooling_layer_2)
        dense_layer = tf.layers.dense(
            flattened_pooling, 1024, activation=tf.nn.relu)
        print(dense_layer.shape)
        
        dropout = tf.layers.dropout(dense_layer, rate=0.4, training=True)
        outputs = tf.layers.dense(dropout, num_classes)
        print(outputs.shape)
        self.choice = tf.argmax(outputs, axis=1)
        self.probability = tf.nn.softmax(outputs)
    

        

Convnet()
