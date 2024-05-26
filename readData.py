import numpy as np
import tensorflow as tf
import glob

# -----------------------------------------------------------

# image_dir = '/root/autodl-tmp/NewDataSet/train_dataset/image/'
# image_dir = '/root/autodl-tmp/New_CCD/image/'
image_dir = '/home/zhouxinlei/ColorConstancy/New_CCD/image/'
# label_dir = '/root/autodl-tmp/NewDataSet/train_dataset/labels2840.npy'
# label_dir = '/root/autodl-tmp/New_CCD/new_labels.npy'
label_dir = '/home/zhouxinlei/ColorConstancy/New_CCD/new_labels.npy'

cpu_count = 64
# cpu_count = multiprocessing.cpu_count()

train_first = 0
train_last = 2272
test_first = 2272
test_last = 2840

# -----------------------------------------------------------

class ReadData:
    def __init__(self):
        self.image_dir = image_dir
        self.image_list = sorted(glob.glob(self.image_dir + '*.png'))
        self.label_dir = label_dir
        self.labels = np.load(self.label_dir)

        self.train_list_length = len(self.image_list[train_first:train_last])
        print('训练集数量：', self.train_list_length)
        self.test_list_length = len(self.image_list[test_first:test_last])
        print('测试集数量：', self.test_list_length)

    def _parse_function(self, filename, label):
        image = tf.read_file(filename)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize_images(image, [224, 224])
        image = tf.cast(image, tf.float32)
        image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
        return image, label

    def read_train_image_label_batch(self, batch_size):
        filenames = tf.constant(self.image_list[train_first:train_last])
        label = tf.constant(self.labels[train_first:train_last], dtype=float)
        datasets = tf.data.Dataset.from_tensor_slices((filenames, label))
        datasets = datasets.map(self._parse_function, num_parallel_calls=cpu_count)
        datasets = datasets.shuffle(buffer_size=(train_last - train_first))
        datasets = datasets.batch(batch_size=batch_size)
        # 使数据集重复，避免iterator.get_next()中断
        datasets = datasets.repeat()
        return datasets

    def read_test_image_label_batch(self, batch_size):
        filenames = tf.constant(self.image_list[test_first:test_last])
        label = tf.constant(self.labels[test_first:test_last], dtype=float)
        datasets = tf.data.Dataset.from_tensor_slices((filenames, label))
        datasets = datasets.map(self._parse_function, num_parallel_calls=cpu_count)
        datasets = datasets.shuffle(buffer_size=(test_last - test_first))
        datasets = datasets.batch(batch_size=batch_size)
        # 使数据集重复，避免iterator.get_next()中断
        datasets = datasets.repeat()
        return datasets
