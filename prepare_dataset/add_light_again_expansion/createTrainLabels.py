import numpy as np

label = np.load('/home/zhouxinlei/ColorConstancy/CCD/labels568.npy')
print('初始labels长度:', len(label))

arr = []

# 复制原有CCD标签
for j in range(len(label)):
    arr.append(label[j])

# 新增标签
for i in range(2272):
    RGB = np.random.randint(0, 255, size=3)
    RGB_l2_normal = np.linalg.norm(RGB)

    R = RGB[0] / RGB_l2_normal
    G = RGB[1] / RGB_l2_normal
    B = RGB[2] / RGB_l2_normal

    new_RGB = np.array([R, G, B])
    arr.append(new_RGB)

    print(arr[i + 568] == new_RGB)

arr = np.array(arr)
print('生成后labels长度:', len(arr))

np.save('./new_labels/labels2840.npy', arr)
np.save('/home/zhouxinlei/ColorConstancy/NewDataSet/train_dataset/labels2840.npy', arr)
