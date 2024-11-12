import cv2 as cv
import numpy as np
#直线检测（霍夫变换） 先要有边缘检测
def line_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)#得到边缘图像
    lines = cv.HoughLines(edges, 1, np.pi/180, 200)
    # 第一个参数是一个二值化图像，所以在进行霍夫变换之前要首先进行二值化，或者进行
    # Canny 边缘检测。第二和第三个值分别代表 ρ 和 θ 的精确度
    # 第四个参数是阈值，检测到的直线的最短长度

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(image, (x1,y1), (x2,y2), (0, 0, 255), 2)
    cv.imshow("image-lines", image)

def line_detect_possible_demo(image):#效果更好
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
    #minLineLength - 线的最短长度。 MaxLineGap - 两条线段之间的最大间隔
    for line in lines:
        #print(type(line))#多维数组
        x1, y1, x2,y2 =line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow("line_detect_possible", image)



src = cv.imread("image/13.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

#line_detection(src)
line_detect_possible_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()
