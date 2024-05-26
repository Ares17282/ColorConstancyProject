import numpy as np
import cv2
import glob

class ReadData:
    def __init__(self):

        # 服务器路径
        self.image_dir = '/root/autodl-tmp/CCD/image/'
        # self.image_dir = '/root/autodl-tmp/NewDataSet/train_dataset/image/'
        # 本地路径
        # self.image_dir = '/home/zhouxinlei/ColorConstancy/NewDataSet/test_dataset/image/'

        self.image_list = sorted(glob.glob(self.image_dir + '*.png'))

        # 服务器路径
        self.label_dir = '/root/autodl-tmp/CCD/labels568.npy'
        # self.label_dir = '/root/autodl-tmp/NewDataSet/train_dataset/labels2840.npy'
        # 本地路径
        # self.label_dir = '/home/zhouxinlei/ColorConstancy/NewDataSet/test_dataset/labels710.npy'

        self.labels = np.load(self.label_dir)

        self.pointer = 0
        self.list_length = len(self.image_list)
        print('训练集总数：', self.list_length)

    def read_image_label_batch(self, batch_size):
        if self.pointer + batch_size < self.list_length:
            first = self.pointer
            last = first + batch_size
            self.pointer = last

        else:
            first = self.list_length - batch_size
            last = first + batch_size
            self.pointer = 0

        image_batch = self.image_list[first:last]
        label_batch = self.labels[first:last]
        images, labels = [], []

        for i in range(batch_size):
            image = self.read_img(image_batch[i])
            image = image[np.newaxis, :, :, :]
            label = label_batch[i, :]
            label = label[np.newaxis, :]

            if i == 0:
                images = image
                labels = label
            else:
                images = np.concatenate((images, image), axis=0)
                labels = np.concatenate((labels, label), axis=0)

        return images, labels

    def read_img(self, image):
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        # 色彩空间变换(BGR->RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        return img / img.max()