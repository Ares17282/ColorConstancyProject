import tensorflow as tf

a = tf.constant([1.2, 2.3, 3.6], shape=[3], name='a')
b = tf.constant([1.2, 2.3, 3.6], shape=[3], name='b')

c = a + b
session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(session.run(c))