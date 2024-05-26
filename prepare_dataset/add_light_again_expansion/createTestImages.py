import numpy as np
import os
import cv2

image_dir = '/home/zhouxinlei/ColorConstancy/NewDataSet/normal_lighting_image/'
image_list = sorted(os.listdir(image_dir))

label = np.load('./new_labels/labels710.npy')
output_dir = '/home/zhouxinlei/ColorConstancy/NewDataSet/test_dataset/image/'

for i in range(710):
    image = np.array(cv2.imread(image_dir + image_list[i % 568]))
    # BGR空间转RGB空间
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    R = float(label[i][0]) / float(np.mean(label[i]))
    G = float(label[i][1]) / float(np.mean(label[i]))
    B = float(label[i][2]) / float(np.mean(label[i]))

    image[:, :, 0] = R * image[:, :, 0]
    image[:, :, 1] = G * image[:, :, 1]
    image[:, :, 2] = B * image[:, :, 2]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if i <= 8:
        cv2.imwrite(output_dir + "img_000%d" % (i + 1) + '.png', image)
        print("img_000%d" % (i + 1) + '.png')
    if i <= 98 and i > 8:
        cv2.imwrite(output_dir + "img_00%d" % (i + 1) + '.png', image)
        print("img_00%d" % (i + 1) + '.png')
    if i <= 998 and i > 98:
        cv2.imwrite(output_dir + "img_0%d" % (i + 1) + '.png', image)
        print("img_0%d" % (i + 1) + '.png')