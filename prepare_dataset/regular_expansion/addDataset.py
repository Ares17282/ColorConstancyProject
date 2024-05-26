import numpy as np
import cv2
import os.path
import random

# 原数据集位置
file_dir = '/home/zhouxinlei/ColorConstancy/CCD/image/'
# 新数据集位置
new_dir = '/home/zhouxinlei/ColorConstancy/New_CCD/image/'
new_dir1 = '/home/zhouxinlei/ColorConstancy/New_CCD/'
# 原标签
old_labels = np.load('/home/zhouxinlei/ColorConstancy/CCD/labels568.npy')
# 新建空白新标签
new_labels = np.empty((0, 3))

# 旋转
def rotate(image,angle,center=None, scale=1.0):
    (h,w) = image.shape[:2]
    if center is None:
        center = (w/2,h/2)
    m = cv2.getRotationMatrix2D(center,angle,scale)
    result = cv2.warpAffine(image, m,(w,h))
    return result

# 翻转
def flip(image):
    result = np.fliplr(image)
    return result

# 如果新数据集存放位置不存在,则新建
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

# 复制原数据集
for img_name in sorted(os.listdir(file_dir)):
    img_path = file_dir + img_name
    img = cv2.imread(img_path)
    cv2.imwrite(new_dir + '0' + img_name[1:6] + '.png',img)
    new_labels = np.vstack([new_labels, old_labels[int(img_name[3:6]) - 1]])
print('1.原数据集复制完成')

# 旋转90
for img_name in sorted(os.listdir(file_dir)):
    img_path = file_dir + img_name
    img = cv2.imread(img_path)
    rotated_90 = rotate(img, 90)
    cv2.imwrite(new_dir + '1' + img_name[1:6] + '.png',rotated_90)
    new_labels = np.vstack([new_labels, old_labels[int(img_name[3:6]) - 1]])
print('2.旋转90度完成')

# 旋转180
for img_name in sorted(os.listdir(file_dir)):
    img_path = file_dir + img_name
    img = cv2.imread(img_path)
    rotated_180 = rotate(img, 180)
    cv2.imwrite(new_dir + '2' + img_name[1:6] + '.png',rotated_180)
    new_labels = np.vstack([new_labels, old_labels[int(img_name[3:6]) - 1]])
print('3.旋转180度完成')

# 水平翻转
for img_name in sorted(os.listdir(file_dir)):
    img_path = file_dir + img_name
    img = cv2.imread(img_path)
    flip1 = flip(img)
    cv2.imwrite(new_dir+'3'+img_name[1:6]+'.png',flip1)
    new_labels = np.vstack([new_labels, old_labels[int(img_name[3:6]) - 1]])
print('4.水平翻转完成')

# 随机切割
for img_name in sorted(os.listdir(file_dir)):
    img_path = file_dir + img_name
    img = cv2.imread(img_path)
    x1 = random.randint(0,100)
    x2 = random.randint(300,500)
    y1 = random.randint(0,300)
    y2 = random.randint(500,800)
    crop = img[x1:x2, y1:y2]
    cv2.imwrite(new_dir+'4'+img_name[1:6]+'.png',crop)
    new_labels = np.vstack([new_labels, old_labels[int(img_name[3:6]) - 1]])
print('5.随机切割完成')

np.save(new_dir1+'new_labels.npy', new_labels)
print('数据集扩充完成!')