import tensorflow as tf
import numpy as np

weights = {
    'conv3_3to64': tf.Variable(tf.random_normal([3, 3, 3, 64])),
    'conv3_256to64': tf.Variable(tf.random_normal([3, 3, 256, 64])),
    'conv5_256to128': tf.Variable(tf.random_normal([5, 5, 256, 128])),
    'conv1_128to64': tf.Variable(tf.random_normal([1, 1, 128, 64])),
    'conv3_64to64': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    'conv1_256to64': tf.Variable(tf.random_normal([1, 1, 256, 64])),
    'conv7_64to64': tf.Variable(tf.random_normal([7, 7, 64, 64])),
    'conv5_64to64': tf.Variable(tf.random_normal([5, 5, 64, 64])),
    'conv1_192to64': tf.Variable(tf.random_normal([1, 1, 192, 64])),
    'conv3_64to3': tf.Variable(tf.random_normal([3, 3, 64, 3])),

    'no_pretrained_1': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'no_pretrained_2': tf.Variable(tf.random_normal([3, 3, 128, 256])),
}

biases = {
    'conv3_3to64': tf.Variable(tf.random_normal([64])),
    'conv3_256to64': tf.Variable(tf.random_normal([64])),
    'conv5_256to128': tf.Variable(tf.random_normal([128])),
    'conv1_128to64': tf.Variable(tf.random_normal([64])),
    'conv3_64to64': tf.Variable(tf.random_normal([64])),
    'conv1_256to64': tf.Variable(tf.random_normal([64])),
    'conv7_64to64': tf.Variable(tf.random_normal([64])),
    'conv5_64to64': tf.Variable(tf.random_normal([64])),
    'conv1_192to64': tf.Variable(tf.random_normal([64])),
    'conv3_64to3': tf.Variable(tf.random_normal([3])),

    'no_pretrained_1': tf.Variable(tf.random_normal([128])),
    'no_pretrained_2': tf.Variable(tf.random_normal([256])),
}


class newNet:
    def __init__(self):
        self.vgg16 = tf.train.NewCheckpointReader('./pretrained_model/vgg_16.ckpt')

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
    def conv_layer(self, input, W, b):
        output = tf.nn.conv2d(input, W, [1, 1, 1, 1], 'SAME')
        output = tf.nn.bias_add(output, b)
        return tf.nn.relu(output)

    # 最大池化
    def max_pooling(self, input, k=2):
        return tf.nn.max_pool(input, [1, k, k, 1], [1, k, k, 1], 'SAME')

    def net(self, input):
        # 224,224,64
        self.conv1 = self.conv_layer(input, weights['conv3_3to64'], biases['conv3_3to64'])
        print('self.conv1:', self.conv1.shape)

        # 112,112,64
        self.maxpool1 = self.max_pooling(self.conv1)
        print('self.maxpool1:', self.maxpool1.shape)

        # 112,112,128
        self.conv2 = self.conv_vgg16(self.maxpool1, 'conv2/conv2_1')
        print('self.conv2:', self.conv2.shape)

        # 112,112,256
        self.conv3 = self.conv_vgg16(self.conv2, 'conv3/conv3_1')
        print('self.conv3:', self.conv3.shape)

        # 不使用预训练模型
        self.no_pretrained_1 = self.conv_layer(self.maxpool1, weights['no_pretrained_1'], biases['no_pretrained_1'])
        self.no_pretrained_2 = self.conv_layer(self.no_pretrained_1, weights['no_pretrained_2'], biases['no_pretrained_2'])

        # 56,56,256
        self.maxpool2 = self.max_pooling(self.conv3)
        print('self.maxpool2:', self.maxpool2.shape)

        # 56,56,64
        self.conv4_1 = self.conv_layer(self.maxpool2, weights['conv3_256to64'], biases['conv3_256to64'])
        print('self.conv4_1:', self.conv4_1.shape)

        # 56,56,128
        self.conv4_2_1 = self.conv_layer(self.maxpool2, weights['conv5_256to128'], biases['conv5_256to128'])
        print('self.conv4_2_1:', self.conv4_2_1.shape)

        # 56,56,64
        self.conv4_2_2 = self.conv_layer(self.conv4_2_1, weights['conv1_128to64'], biases['conv1_128to64'])
        print('self.conv4_2_2:', self.conv4_2_2.shape)

        # 56,56,64
        self.conv4_2_3 = self.conv_layer(self.conv4_2_2, weights['conv3_64to64'], biases['conv3_64to64'])
        print('self.conv4_2_3:', self.conv4_2_3.shape)

        # 56,56,64
        self.conv4_3_1 = self.conv_layer(self.maxpool2, weights['conv1_256to64'], biases['conv1_256to64'])
        print('self.conv4_3_1:', self.conv4_3_1.shape)

        # 56,56,64
        self.conv4_3_2 = self.conv_layer(self.conv4_3_1, weights['conv7_64to64'], biases['conv7_64to64'])
        print('self.conv4_3_2:', self.conv4_3_2.shape)

        # 56,56,64
        self.conv4_3_3 = self.conv_layer(self.conv4_3_2, weights['conv5_64to64'], biases['conv5_64to64'])
        print('self.conv4_3_3:', self.conv4_3_3.shape)

        # 56,56,192(特征融合)
        self.conv5 = tf.concat([self.conv4_1, self.conv4_2_3, self.conv4_3_3], axis=3)
        print('self.conv5:', self.conv5.shape)

        # 28,28,192
        self.maxpool3 = self.max_pooling(self.conv5)
        print('self.maxpool3:', self.maxpool3.shape)

        # 28,28,64
        self.conv6 = self.conv_layer(self.maxpool3, weights['conv1_192to64'], biases['conv1_192to64'])
        print('self.conv6:', self.conv6.shape)

        # 14,14,64
        self.maxpool4 = self.max_pooling(self.conv6)
        print('self.maxpool4:', self.maxpool4.shape)

        # 14,14,3
        self.conv7 = self.conv_layer(self.maxpool4, weights['conv3_64to3'], biases['conv3_64to3'])
        print('self.conv7:', self.conv7.shape)

        # 7,7,3
        self.maxpool5 = self.max_pooling(self.conv7)
        print('self.maxpool5:', self.maxpool5.shape)

        # 3(通道合并)
        self.result = tf.reduce_sum(self.maxpool5, axis=(1, 2))
        print('self.result:', self.result.shape)

        return tf.nn.l2_normalize(self.result)

    def lossFunction(self, logits, labels):
        # # 角度损失函数
        safe_v = tf.constant(0.999999)
        dot = tf.reduce_sum(logits * labels, axis=1)
        dot = tf.clip_by_value(dot, -safe_v, safe_v)
        loss = tf.acos(dot) * (180 / np.pi)
        loss = tf.reduce_mean(loss)

        # loss = tf.reduce_sum(tf.square(logits - labels))

        return loss
