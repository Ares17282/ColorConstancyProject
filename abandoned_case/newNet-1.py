import tensorflow as tf
import numpy as np

weights = {
    'conv1': tf.Variable(tf.random_normal([3, 3, 3, 16])),
    'conv2': tf.Variable(tf.random_normal([3, 3, 16, 64])),
    'conv5_1': tf.Variable(tf.random_normal([3, 3, 256, 64])),
    'conv5_2_1': tf.Variable(tf.random_normal([5, 5, 256, 128])),
    'conv5_2_2': tf.Variable(tf.random_normal([3, 3, 128, 64])),
    'conv5_2_3': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    'conv5_3_1': tf.Variable(tf.random_normal([7, 7, 256, 64])),
    'conv5_3_2': tf.Variable(tf.random_normal([5, 5, 64, 64])),
    'conv5_3_3': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    'conv7': tf.Variable(tf.random_normal([3, 3, 192, 64])),
    'conv8': tf.Variable(tf.random_normal([3, 3, 64, 16])),
    'conv9': tf.Variable(tf.random_normal([3, 3, 16, 3])),
}

biases = {
    'conv1': tf.Variable(tf.random_normal([16])),
    'conv2': tf.Variable(tf.random_normal([64])),
    'conv5_1': tf.Variable(tf.random_normal([64])),
    'conv5_2_1': tf.Variable(tf.random_normal([128])),
    'conv5_2_2': tf.Variable(tf.random_normal([64])),
    'conv5_2_3': tf.Variable(tf.random_normal([64])),
    'conv5_3_1': tf.Variable(tf.random_normal([64])),
    'conv5_3_2': tf.Variable(tf.random_normal([64])),
    'conv5_3_3': tf.Variable(tf.random_normal([64])),
    'conv7': tf.Variable(tf.random_normal([64])),
    'conv8': tf.Variable(tf.random_normal([16])),
    'conv9': tf.Variable(tf.random_normal([3])),
}


class newNet:
    def __init__(self):
        self.vgg16 = tf.train.NewCheckpointReader('./pretrained_model/vgg_16.ckpt')

    def net(self, input):
        self.conv1 = self.conv_layer(input, weights['conv1'], biases['conv1'], 2)
        print('self.conv1:', self.conv1.shape)  # 112,112,16

        self.conv2 = self.conv_layer(self.conv1, weights['conv2'], biases['conv2'], 2)
        print('self.conv2:', self.conv2.shape)  # 56,56,64

        self.conv3 = self.conv_vgg16(self.conv2, 'conv2/conv2_1')
        print('self.conv3:', self.conv3.shape)  # 56,56,128

        self.conv4 = self.conv_vgg16(self.conv3, 'conv3/conv3_1')
        print('self.conv4:', self.conv4.shape)  # 56,56,256

        self.conv5_1 = self.conv_layer(self.conv4, weights['conv5_1'], biases['conv5_1'], 2)
        print('self.conv5_1:', self.conv5_1.shape)  # 28,28,64

        self.conv5_2_1 = self.conv_layer(self.conv4, weights['conv5_2_1'], biases['conv5_2_1'], 1)
        print('self.conv5_2_1:', self.conv5_2_1.shape)   # 56,56,128

        self.conv5_2_2 = self.conv_layer(self.conv5_2_1, weights['conv5_2_2'], biases['conv5_2_2'], 1)
        print('self.conv5_2_2:', self.conv5_2_2.shape)  # 56,56,64

        self.conv5_2_3 = self.conv_layer(self.conv5_2_2, weights['conv5_2_3'], biases['conv5_2_3'], 2)
        print('self.conv5_2_3:', self.conv5_2_3.shape)  # 28,28,64

        self.conv5_3_1 = self.conv_layer(self.conv4, weights['conv5_3_1'], biases['conv5_3_1'], 1)
        print('self.conv5_3_1:', self.conv5_3_1.shape)  # 56,56,64

        self.conv5_3_2 = self.conv_layer(self.conv5_3_1, weights['conv5_3_2'], biases['conv5_3_2'], 1)
        print('self.conv5_3_2:', self.conv5_3_2.shape)  # 56,56,64

        self.conv5_3_3 = self.conv_layer(self.conv5_3_2, weights['conv5_3_3'], biases['conv5_3_3'], 2)
        print('self.conv5_3_3:', self.conv5_3_3.shape)  # 28,28,64


        self.conv6 = tf.concat([self.conv5_1, self.conv5_2_3, self.conv5_3_3], axis=3)
        print('self.conv6:', self.conv6.shape)   # 28,28,192(特征融合)

        self.conv7 = self.conv_layer(self.conv6, weights['conv7'], biases['conv7'], 2)
        print('self.conv7:', self.conv7.shape)  # 14,14,64

        self.conv8 = self.conv_layer(self.conv7, weights['conv8'], biases['conv8'], 2)
        print('self.conv8:', self.conv8.shape)  # 7,7,16

        self.conv9 = self.conv_layer(self.conv8, weights['conv9'], biases['conv9'], 2)
        print('self.conv9:', self.conv9.shape)  # 4,4,3

        self.result = tf.reduce_sum(self.conv9, axis=(1, 2))
        print('self.result:', self.result.shape)

        return tf.nn.l2_normalize(self.result, axis=1)


    def lossFunction(self, logits, labels):
        # # 角度损失函数
        safe_v = tf.constant(0.999999)
        dot = tf.reduce_sum(logits * labels, axis=1)
        dot = tf.clip_by_value(dot, -safe_v, safe_v)
        loss = tf.acos(dot) * (180 / np.pi)
        loss = tf.reduce_mean(loss)

        # loss = tf.reduce_sum(tf.square(logits - labels))
        return loss

    # 获取vgg16的weights
    def get_vgg16_weights(self, path):
        return self.vgg16.get_tensor('vgg_16/' + path + '/weights')

    # 获取vgg16的biases
    def get_vgg16_biases(self, path):
        return self.vgg16.get_tensor('vgg_16/' + path + '/biases')

    # 重建vgg16卷积过程
    def conv_vgg16(self, input, path):
        W = self.get_vgg16_weights(path)
        b = self.get_vgg16_biases(path)
        conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], 'SAME')
        conv = tf.nn.bias_add(conv, b)
        return tf.nn.relu(conv)

    # 卷积层
    def conv_layer(self, input, W, b, strides):
        output = tf.nn.conv2d(input, W, [1, strides, strides, 1], 'SAME')
        output = tf.nn.bias_add(output, b)
        return tf.nn.relu(output)

    # 最大池化
    def max_pooling(self, input, k=2):
        return tf.nn.max_pool(input, [1, k, k, 1], [1, k, k, 1], 'SAME')
