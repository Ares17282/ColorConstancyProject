import cv2
import numpy as np

def gamma_correction(image, gamma):
    # 对图像进行伽马校正
    corrected_image = np.uint8(cv2.pow(image / 255.0, gamma) * 255)

    return corrected_image

def gamma(original_image, image):
    max_brightness = np.mean(image)
    target_gamma = 1.0 / 2.2

    # 计算校正参数
    gamma = np.log(target_gamma) / np.log(max_brightness / 255.0)

    # 对图像进行伽马校正
    corrected_image = gamma_correction(image, gamma)

    # 显示原始图像和校正后的图像
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original Image', 800, 600)

    cv2.namedWindow('Corrected Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Corrected Image', 800, 600)

    cv2.imshow('Original Image', cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    cv2.imshow('Corrected Image', cv2.cvtColor(corrected_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()