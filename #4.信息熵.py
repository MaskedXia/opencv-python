import cv2
import numpy as np
import math
import time

def get_Std(img):
    m, s = cv2.meanStdDev(img)
    ave_std = sum(m)/len(m)
    print("标准差为： ", ave_std)
    return ave_std

def get_entropy(img_):
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    x, y = img_.shape[0:2]
    img_ = cv2.resize(img_, (100, 100)) # 缩小的目的是加快计算速度
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0
    img = np.array(img_)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k =  float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if(tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    print("信息熵为： ", res)
    return res

def getImageVar(image):

    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()

    print("清晰度为： ", imageVar)

    return imageVar




if __name__ == '__main__':
    img = 'insulator1/001/MSRCP3546.jpg'
    image = cv2.imread(img)

    std = get_Std(image)
    res = get_entropy(image)
    cle = getImageVar(image)


    cv2.waitKey(0)
    cv2.destroyAllWindows()