import numpy as np

data = np.load('/home/zhouxinlei/ColorConstancy/CCD/labels568.npy')
data1 = np.load('/home/zhouxinlei/ColorConstancy/New_CCD/new_labels.npy')

for i,v in enumerate(data1):
    n = i % 568
    if (data[n] != data1[i]).all():
        print('数据集标签有问题!')
        break
    else:
        if i == 2839:
            print('数据集没有问题!')