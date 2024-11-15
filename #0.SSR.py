import numpy as np
import cv2
import math


def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

def SSR(src_img, size):
    L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)

    dst_Img = cv2.log(img/255.0)
    dst_Lblur = cv2.log(L_blur/255.0)
    dst_IxL = cv2.multiply(dst_Img,dst_Lblur)
    log_R = cv2.subtract(dst_Img, dst_IxL)
    #log_R = math.e ** (log_R)

    dst_R = cv2.normalize(log_R,None,0,255,cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    #log_uint8 = math.e**(log_uint8)
    return log_uint8
'''void cv::convertScaleAbs(  对图像数组每一个元素做如下操作
	cv::InputArray src, // 输入数组
	cv::OutputArray dst, // 输出数组
	double alpha = 1.0, // 乘数因子
	double beta = 0.0 // 偏移量
'''


if __name__ == '__main__':
    img = 'insulator1/1072.jpg'
    size = 3
    src_img = cv2.imread(img)
    b_gray, g_gray, r_gray = cv2.split(src_img)
    b_gray = SSR(b_gray, size)
    g_gray = SSR(g_gray, size)
    r_gray = SSR(r_gray, size)
    result = cv2.merge([b_gray, g_gray, r_gray])

    cv2.imshow('img',src_img)
    cv2.imshow('result',result)
    cv2.imwrite('insulator1/SSR1072.jpg', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

