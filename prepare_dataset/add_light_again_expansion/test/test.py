import numpy as np
import os
import cv2

image_dir = '/home/zhouxinlei/ColorConstancy/NewDataSet/train_dataset/image/'
label = np.load('../new_labels/labels710.npy')
output_dir = '../train_result/'

i = 65

image_name = 'img_00' + str(i) + '.png'

image = np.array(cv2.imread(image_dir + image_name))

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

R = float(np.mean(label[i-1])) / float(label[i-1][0])
G = float(np.mean(label[i-1])) / float(label[i-1][1])
B = float(np.mean(label[i-1])) / float(label[i-1][2])

image[:, :, 0] = R * image[:, :, 0]
image[:, :, 1] = G * image[:, :, 1]
image[:, :, 2] = B * image[:, :, 2]

image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

cv2.imwrite(output_dir + 'test_' + str(i) + '.png', image)
