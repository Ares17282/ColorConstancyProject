import numpy as np
import os
import cv2

image_dir = '/home/zhouxinlei/ColorConstancy/CCD/image/'
label = np.load('/home/zhouxinlei/ColorConstancy/CCD/labels568.npy')
output_dir = '/home/zhouxinlei/ColorConstancy/NewDataSet/normal_lighting_image/'

image_list = sorted(os.listdir(image_dir))
# print(image_list)

for i in range(568):
    image = np.array(cv2.imread(image_dir + image_list[i]))
    # BGR空间转RGB空间
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    R = float(np.mean(label[i])) / float(label[i][0])
    G = float(np.mean(label[i])) / float(label[i][1])
    B = float(np.mean(label[i])) / float(label[i][2])

    image[:, :, 0] = R * image[:, :, 0]
    image[:, :, 1] = G * image[:, :, 1]
    image[:, :, 2] = B * image[:, :, 2]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if i <= 8:
        cv2.imwrite(output_dir + "img_000%d" % (i + 1) + '.png', image)
        print("img_000%d" % (i + 1) + '.png')
    if i <=98 and i > 8:
        cv2.imwrite(output_dir + "img_00%d" % (i + 1) + '.png', image)
        print("img_00%d" % (i + 1) + '.png')
    if i <=998 and i> 98:
        cv2.imwrite(output_dir + "img_0%d" % (i + 1) + '.png', image)
        print("img_0%d" % (i + 1) + '.png')
