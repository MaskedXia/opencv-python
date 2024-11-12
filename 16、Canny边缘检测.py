import cv2 as cv#Canny边缘提取
import numpy as np
'''
1.高斯模糊 GaussianBlur
2.灰度转换 cvtColor
3.计算梯度 sobel/Scharr
4.非最大信号抑制 （局部最大）
5.高低阈值输出二值图像 高于T2保留，低于T1丢弃，中间相互连接的就保留 T2：T1 =3/1或2/1
'''
def edge_demo(image):
    blurred = cv.GaussianBlur(image, (3,3), 0,)#可以降噪声
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    grad_x = cv.Sobel(image, cv.CV_16SC1, 1, 0)  # Sobel可以转换为Scharr，效果更强
    grad_y = cv.Sobel(image, cv.CV_16SC1, 0, 1)
    #edge_output = cv.Canny(grad_x, grad_y, 50, 150)
    edge_output = cv.Canny(blurred, 50, 150)
    cv.imshow("Canny Edge", edge_output)

    dst = cv.bitwise_and(image, image, mask = edge_output)
    cv.imshow("Colored Edge", dst)



src = cv.imread("image/1.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

edge_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()
