import numpy as np

arr = []

# 新增标签
for i in range(710):
    RGB = np.random.randint(0, 255, size=3)
    RGB_l2_normal = np.linalg.norm(RGB)

    R = RGB[0] / RGB_l2_normal
    G = RGB[1] / RGB_l2_normal
    B = RGB[2] / RGB_l2_normal

    new_RGB = np.array([R, G, B])
    arr.append(new_RGB)

    print(arr[i] == new_RGB)

arr = np.array(arr)
print('生成后labels长度:', len(arr))

np.save('./new_labels/labels710.npy', arr)
np.save('/home/zhouxinlei/ColorConstancy/NewDataSet/test_dataset/labels710.npy', arr)
