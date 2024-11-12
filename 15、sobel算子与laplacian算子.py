import cv2 as cv
import numpy as np

#图像梯度（一阶导数与Sobel算子  二阶导数与拉普拉斯算子）
def sobel_demo(image):
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)#Sobel可以转换为Scharr，效果更强
    # cv2.CV_32F 输出图像的深度（数据类型），可以使用 -1, 与原图像保持一致 np.uint8
    #
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)
    gradx = cv.convertScaleAbs(grad_x)#取绝对值,导数可能是负数
    #或者gradx = np.absolute(gradx)
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("gradient_x", gradx)
    #cv.imshow("try_x", grad_x)
    cv.imshow("gradient_y", grady)

    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.imshow("gradient", gradxy)

def laplace_demo(image):
    #dst = cv.Laplacian(image, cv.CV_32F)#默认4邻域
    #lpls = cv.convertScaleAbs(dst)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])#也可以[[1, 1, 1], [1, -8, 1], [1, 1, 1]]8邻域
    dst = cv.filter2D(image, cv.CV_32F, kernel)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow("laplace", lpls)

src = cv.imread("image/5.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

sobel_demo(src)
#laplace_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()
