import tensorflow as tf
import numpy as np


class newNet:
    def conv_layer_relu(self, input, W, b, strides):
        output = tf.nn.conv2d(input, W, [1, strides, strides, 1], 'SAME')
        output = tf.nn.bias_add(output, b)
        return tf.nn.relu(output)

    def conv_layer(self, input, W, b, strides):
        output = tf.nn.conv2d(input, W, [1, strides, strides, 1], 'SAME')
        output = tf.nn.bias_add(output, b)
        return output

    def conv_vgg16(self, input, path):
        vgg16 = tf.train.NewCheckpointReader('./pretrained_model/vgg_16.ckpt')
        W = vgg16.get_tensor('vgg_16/' + path + '/weights')
        b = vgg16.get_tensor('vgg_16/' + path + '/biases')
        conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], 'SAME')
        conv = tf.nn.bias_add(conv, b)
        return tf.nn.relu(conv)

    def max_pooling(self, input, k=2):
        return tf.nn.max_pool(input, [1, k, k, 1], [1, k, k, 1], 'SAME')

    def weight(self, name, shape):
        return tf.get_variable('W_' + name, shape)

    def biase(self, name, shape):
        return tf.get_variable('b_' + name, shape)

    def net_module(self, input, i):
        conv1_1 = self.conv_layer_relu(input, self.weight(str(i) + 'conv1_1', [3, 3, 64, 64]), self.biase(str(i) + 'conv1_1', [64]), 1)
        conv1_1 = self.max_pooling(conv1_1)
        conv1_2 = self.conv_layer(conv1_1, self.weight(str(i) + 'conv1_2', [1, 1, 64, 64]), self.biase(str(i) + 'conv1_2', [64]), 1)
        conv1_3 = self.conv_layer_relu(conv1_2, self.weight(str(i) + 'conv1_3', [3, 3, 64, 64]), self.biase(str(i) + 'conv1_3', [64]), 1)
        conv1_3 = self.max_pooling(conv1_3)

        conv2_1 = self.conv_layer_relu(input, self.weight(str(i) + 'conv2_1', [5, 5, 64, 128]), self.biase(str(i) + 'conv2_1', [128]), 1)
        conv2_2 = self.conv_layer_relu(conv2_1, self.weight(str(i) + 'conv2_2', [3, 3, 128, 64]), self.biase(str(i) + 'conv2_2', [64]), 1)
        conv2_2 = self.max_pooling(conv2_2)
        conv2_3 = self.conv_layer(conv2_2, self.weight(str(i) + 'conv2_3', [1, 1, 64, 64]), self.biase(str(i) + 'conv2_3', [64]), 1)
        conv2_4 = self.conv_layer_relu(conv2_3, self.weight(str(i) + 'conv2_4', [3, 3, 64, 64]), self.biase(str(i) + 'conv2_4', [64]), 1)
        conv2_4 = self.max_pooling(conv2_4)

        conv3_1 = self.conv_layer_relu(input, self.weight(str(i) + 'conv3_1', [7, 7, 64, 128]), self.biase(str(i) + 'conv3_1', [128]), 1)
        conv3_2 = self.conv_layer_relu(conv3_1, self.weight(str(i) + 'conv3_2', [5, 5, 128, 128]), self.biase(str(i) + 'conv3_2', [128]), 1)
        conv3_3 = self.conv_layer_relu(conv3_2, self.weight(str(i) + 'conv3_3', [3, 3, 128, 64]), self.biase(str(i) + 'conv3_3', [64]), 1)
        conv3_3 = self.max_pooling(conv3_3)
        conv3_4 = self.conv_layer(conv3_3, self.weight(str(i) + 'conv3_4', [1, 1, 64, 64]), self.biase(str(i) + 'conv3_4', [64]), 1)
        conv3_5 = self.conv_layer_relu(conv3_4, self.weight(str(i) + 'conv3_5', [5, 5, 64, 64]), self.biase(str(i) + 'conv3_5', [64]), 1)
        conv3_6 = self.conv_layer_relu(conv3_5, self.weight(str(i) + 'conv3_6', [3, 3, 64, 64]), self.biase(str(i) + 'conv3_6', [64]), 1)
        conv3_6 = self.max_pooling(conv3_6)

        conv6 = tf.concat([conv1_3, conv2_4, conv3_6], axis=3)
        conv6 = self.conv_layer(conv6, self.weight(str(i) + 'conv6', [1, 1, 192, 64]), self.biase(str(i) + 'conv6', [64]), 1)

        return conv6

    def net(self, input):
        net1 = self.conv_vgg16(input, 'conv1/conv1_1')
        net1 = self.max_pooling(net1)
        net1 = self.conv_vgg16(net1, 'conv1/conv1_2')

        net2_1 = self.net_module(net1, 1)
        net2_2 = self.net_module(net1, 2)
        net2 = tf.add(net2_1, net2_2)
        net2 = tf.nn.relu(net2)

        net3_1 = self.net_module(net2, 3)
        net3_2 = self.net_module(net2, 4)
        net3 = tf.add(net3_1, net3_2)
        net3 = tf.nn.relu(net3)

        net1 = tf.image.resize_bilinear(net1, [7, 7])
        net2 = tf.image.resize_bilinear(net2, [7, 7])
        net5 = tf.concat([net1, net2, net3], axis=3)
        net5 = tf.nn.relu(net5)

        net6 = self.conv_layer_relu(net5, self.weight('net6', [5, 5, 192, 128]), self.biase('net6', [128]), 1)
        net6 = self.conv_layer_relu(net6, self.weight('net6_1', [3, 3, 128, 64]), self.biase('net6_1', [64]), 2)
        net6 = tf.nn.dropout(net6, 0.6)
        net7 = self.conv_layer(net6, self.weight('net7', [3, 3, 64, 4]), self.biase('net7', [4]), 2)
        net7 = self.max_pooling(net7)

        confidence_layer = net7[:, :, :, 0]
        rgb_channels = net7[:, :, :, 1:]
        net7 = confidence_layer * rgb_channels
        result = tf.reduce_sum(net7, axis=(1, 2))

        return tf.nn.l2_normalize(result, axis=1)

    def lossFunction(self, logits, labels):
        # 角度损失函数
        safe_v = tf.constant(0.999999)
        dot = tf.reduce_sum(logits * labels, axis=1)
        dot = tf.clip_by_value(dot, -safe_v, safe_v)
        loss = tf.acos(dot) * (180.0 / np.pi)
        loss = tf.reduce_mean(loss)
        return loss