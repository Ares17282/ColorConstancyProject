import tensorflow as tf
import readData
import newNet
import numpy as np
import os
# 仅显示错误信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -----------------------------------------------------------

learning_rate = 0.0001
batch_size = 16
iteration_steps = 15000
update_step = 300
update_multiple = 0.9

max_steps_without_improvement = 2000
best_loss = float('inf')
steps_without_improvement = 0

# -----------------------------------------------------------

img_X = tf.placeholder(tf.float32, [None, 224, 224, 3])
lab_Y = tf.placeholder(tf.float32, [None, 3])

Data = readData.ReadData()
Model = newNet.newNet()

net_output = Model.net(img_X)
loss_op = Model.lossFunction(logits=net_output, labels=lab_Y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss=loss_op)

save = tf.train.Saver()

tendency = []

best_loss_step = -1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_iterator = Data.read_train_image_label_batch(batch_size=batch_size).make_initializable_iterator()
    test_iterator = Data.read_test_image_label_batch(batch_size=batch_size).make_initializable_iterator()
    sess.run(train_iterator.initializer)
    sess.run(test_iterator.initializer)
    train_next_element = train_iterator.get_next()
    test_next_element = test_iterator.get_next()
    for step in range(iteration_steps):

        if step % update_step == 0 and learning_rate > 0.000005:
            learning_rate = update_multiple * learning_rate

        train_img_batch, train_lab_batch = sess.run(train_next_element)

        _, loss = sess.run([train_op, loss_op], feed_dict={img_X: train_img_batch, lab_Y: train_lab_batch})

        print('Step' + str(step+1) + ',Minibatch Loss=' + '{:.4f}'.format(loss)
              + '-----Best Loss: Step' + str(best_loss_step+1) + ',Best Loss=' + '{:.4f}'.format(best_loss))

        coordinate = np.array([step, '{:.4f}'.format(loss)])
        tendency.append(coordinate)

        test_img_batch, test_lab_batch = sess.run(test_next_element)
        val_loss = sess.run(loss_op, feed_dict={img_X: test_img_batch, lab_Y: test_lab_batch})

        if val_loss < best_loss:
            # 如果验证集性能提升，保存最佳模型
            if best_loss == float('inf'):
                best_loss = val_loss
            else:
                best_loss = (val_loss + best_loss)/2

            best_loss_step = step
            save.save(sess, './resultModel/model.ckpt')
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
            if steps_without_improvement > max_steps_without_improvement:
                # 如果连续若干步没有验证集性能提升，停止训练
                print("No improvement in validation loss for {} steps. Stopping training.".format(
                    max_steps_without_improvement))
                break

    print('Optimization Finished!')

    save.save(sess, './resultModel_1/model.ckpt')

    np.save('resultModel/tendency_train.npy', tendency)
