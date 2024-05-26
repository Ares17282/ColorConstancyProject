import tensorflow as tf
import numpy as np
import newNet
import gamma
import os
# 仅显示错误信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

image_path = '/home/zhouxinlei/ColorConstancy/New_CCD/image/000058.png'

image = tf.read_file(image_path)
image = tf.image.decode_png(image, channels=3)
image = tf.image.resize_images(image, [224, 224])
image = tf.cast(image, tf.float32)
image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))

new_image = tf.read_file(image_path)
new_image = tf.image.decode_png(new_image, channels=3)
original_image = tf.read_file(image_path)
original_image = tf.image.decode_png(original_image, channels=3)

X = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])

model = newNet.newNet()
netOut = model.net(X)

with tf.Session() as sess:
    save = tf.train.Saver()
    save.restore(sess, './resultModel_final/1/model.ckpt')

    image_np = sess.run(image)
    original_image = sess.run(original_image)
    new_image = sess.run(new_image)

    feed_dict = {X: [image_np]}
    net_output = sess.run(netOut, feed_dict)
    net_output = net_output[0]
    mean_net = tf.reduce_mean(net_output)
    print('预测的真实光照为:', net_output)

    R = float(np.max(net_output)) / float(net_output[0])
    G = float(np.max(net_output)) / float(net_output[1])
    B = float(np.max(net_output)) / float(net_output[2])

    new_image[:, :, 0] = np.minimum(R * new_image[:, :, 0], 255).astype(np.uint8)
    new_image[:, :, 1] = np.minimum(G * new_image[:, :, 1], 255).astype(np.uint8)
    new_image[:, :, 2] = np.minimum(B * new_image[:, :, 2], 255).astype(np.uint8)

    gamma.gamma(original_image, new_image)
