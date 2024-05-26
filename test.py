import tensorflow as tf
import numpy as np
import newNet
import readData
import os
# 仅显示错误信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 1
iteration_steps = 2840

X = tf.placeholder(tf.float32, [None, 224, 224, 3])
Y = tf.placeholder(tf.float32, [None, 3])

model = newNet.newNet()
netOut = model.net(X)

Data = readData.ReadData()
iterator = Data.read_test_image_label_batch(batch_size=batch_size).make_initializable_iterator()

tendency = []

with tf.Session() as sess2:
    # sess2.run(tf.global_variables_initializer())
    sess2.run(iterator.initializer)
    next_element = iterator.get_next()
    save = tf.train.Saver()
    save.restore(sess2, './resultModel_final/1/model.ckpt')

    for step in range(iteration_steps):
        img_batch, lab_batch = sess2.run(next_element)
        feed_dict = {X: img_batch}
        net_output = sess2.run(netOut, feed_dict)

        loss = sess2.run(model.lossFunction(net_output, lab_batch))

        print('Step' + str(step+1) + ',Mean Loss=' + '{:.4f}'.format(loss))

        coordinate = np.array([step, '{:.4f}'.format(loss)])
        tendency.append(coordinate)

    loss_values = [float(x[1]) for x in tendency]
    mean_loss = sum(loss_values) / len(loss_values)
    print("Mean Loss:", mean_loss)
    np.save('resultModel_final/1/tendency_test.npy', tendency)
