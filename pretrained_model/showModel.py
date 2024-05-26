import tensorflow as tf
# 使用pprint 提高打印的可读性
import pprint
NewCheck = tf.train.NewCheckpointReader('./vgg_16.ckpt')

print("打印网络结构如下:")
# 类型是str
pprint.pprint(NewCheck.debug_string().decode("utf-8"))

# 获取指定层数到weights和biases
print(NewCheck.get_tensor('vgg_16/fc7/biases'))


# 打印网络结构如下:
#  (
#  'vgg_16/fc8/weights (DT_FLOAT) [1,1,4096,1000]\n'
#  'vgg_16/fc7/weights (DT_FLOAT) [1,1,4096,4096]\n'
#  'vgg_16/fc7/biases (DT_FLOAT) [4096]\n'
#  'vgg_16/fc6/weights (DT_FLOAT) [7,7,512,4096]\n'
#  'vgg_16/conv3/conv3_1/weights (DT_FLOAT) [3,3,128,256]\n'
#  'vgg_16/fc8/biases (DT_FLOAT) [1000]\n'
#  'vgg_16/conv5/conv5_2/biases (DT_FLOAT) [512]\n'
#  'vgg_16/conv3/conv3_1/biases (DT_FLOAT) [256]\n'
#  'vgg_16/conv4/conv4_3/weights (DT_FLOAT) [3,3,512,512]\n'
#  'vgg_16/conv2/conv2_1/weights (DT_FLOAT) [3,3,64,128]\n'
#  'vgg_16/conv1/conv1_1/biases (DT_FLOAT) [64]\n'
#  'vgg_16/conv5/conv5_3/weights (DT_FLOAT) [3,3,512,512]\n'
#  'vgg_16/conv5/conv5_3/biases (DT_FLOAT) [512]\n'
#  'vgg_16/conv2/conv2_2/biases (DT_FLOAT) [128]\n'
#  'vgg_16/conv1/conv1_2/weights (DT_FLOAT) [3,3,64,64]\n'
#  'vgg_16/conv2/conv2_1/biases (DT_FLOAT) [128]\n'
#  'vgg_16/conv2/conv2_2/weights (DT_FLOAT) [3,3,128,128]\n'
#  'vgg_16/conv1/conv1_2/biases (DT_FLOAT) [64]\n'
#  'vgg_16/conv1/conv1_1/weights (DT_FLOAT) [3,3,3,64]\n'
#  'vgg_16/conv3/conv3_2/biases (DT_FLOAT) [256]\n'
#  'vgg_16/conv5/conv5_1/weights (DT_FLOAT) [3,3,512,512]\n'
#  'vgg_16/conv3/conv3_2/weights (DT_FLOAT) [3,3,256,256]\n'
#  'vgg_16/conv3/conv3_3/biases (DT_FLOAT) [256]\n'
#  'vgg_16/fc6/biases (DT_FLOAT) [4096]\n'
#  'vgg_16/conv5/conv5_2/weights (DT_FLOAT) [3,3,512,512]\n'
#  'vgg_16/conv3/conv3_3/weights (DT_FLOAT) [3,3,256,256]\n'
#  'vgg_16/mean_rgb (DT_FLOAT) [3]\n'
#  'vgg_16/conv4/conv4_1/biases (DT_FLOAT) [512]\n'
#  'global_step (DT_INT64) []\n'
#  'vgg_16/conv4/conv4_1/weights (DT_FLOAT) [3,3,256,512]\n'
#  'vgg_16/conv4/conv4_2/biases (DT_FLOAT) [512]\n'
#  'vgg_16/conv4/conv4_2/weights (DT_FLOAT) [3,3,512,512]\n'
#  'vgg_16/conv4/conv4_3/biases (DT_FLOAT) [512]\n'
#  'vgg_16/conv5/conv5_1/biases (DT_FLOAT) [512]\n'
#  )