import cv2 as cv
import numpy as np
#模糊操作（卷积）   均值模糊（高斯模糊）、中值模糊、自定义模糊
# 我们也可以对 2D 图像实施低通滤波（LPF），高通滤波(HPF)等。LPF
# 帮助我们去除噪音，模糊图像。HPF 帮助我们找到图像的边缘

def blur_demo(image):#均值模糊
    dst = cv.blur(image, (1, 15))#1代表x方向模糊程度，15代表y方向模糊程度
    cv.imshow("blur", dst)

def clamp(pv):
    if pv > 255:
        return 255
    if pv < 0:
        return 0
    else:
        return pv

def gaussian_noise(image):#高斯噪声
    h, w, c =image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 100, 3)#均值0，方差20，size3的随机数
            b = image[row, col, 0]#blue
            g = image[row, col, 1]#green
            r = image[row, col, 2]#red
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    cv.imshow("noise image", image)
    cv.imwrite("insulator1/3145.jpg",image)
    dst = cv.GaussianBlur(src, (3,3), 0)
    #高斯模糊，处理高斯噪声，高斯核的宽和高（必须是奇数）,0是x方向标准差
    cv.imshow("GaussianBlur", dst)
    cv.imwrite("insulator1/31455.jpg", dst)

def medium_blur_demo(image):#中值模糊
    dst = cv.medianBlur(image, 5)
    cv.imshow("medium_blur", dst)#处理椒盐噪声

def custom_blur_demo(image):#除此外还有双边滤波
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]], np.float32)/9
    #卷积核,必须奇数，实现锐化
    #kernel = np.ones([5, 5], np.float32)/25
    dst = cv.filter2D(image, -1, kernel= kernel)
    ##-1:目标图像的所需深度
    cv.imshow("custom_blur", dst)

src = cv.imread("insulator/3145.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

#filter(src)
# blur_demo(src)
# medium_blur_demo(src)
# custom_blur_demo(src)

# dst = cv.GaussianBlur(src, (5,5), 15)#高斯模糊
# cv.imshow("GaussianBlur", dst)

t1 = cv.getTickCount()
gaussian_noise(src)
t2 = cv.getTickCount()
time = (t2 - t1)/cv.getTickFrequency()
print(time)



cv.waitKey(0)
cv.destroyAllWindows()