import cv2 as cv
import numpy as np
#弧长和面积，多边形拟合
def measure_object(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("threshold value", ret)
    cv.imshow("binary", binary)
    dst = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    cloneImage, contours, heriachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)#轮廓面积
        print("contours area", area)
        #perimeter = cv.arcLength(contour, True)#计算弧长， True表示闭合
        x, y, w, h = cv.boundingRect(contour)#轮廓的外接矩形 , 旋转外接矩形见110
        rate = min(w, h)/max(w, h)
        print("rectangle rate", rate)
        mm = cv.moments(contour)#字典几何矩
        cx = mm['m10'] / mm['m00']#x方向重心
        cy = mm['m01'] / mm['m00']#y方向重心
        cv.circle(dst, (np.int(cx), np.int(cy)), 3, (0, 255, 255), -1)#画圆，画点
        #cv.rectangle(dst, (x, y), (x+w, y+h), (255, 255, 0), 2)#画矩形
        approxCurve = cv.approxPolyDP(contour, 4, True)#轮廓近似
        #使用Douglas-Peucker算法，二个参数叫epsilon，它是从原始轮廓到近似轮廓的最大距离，第三个参数设定弧线是否闭合
        #返回近似轮廓
        print(approxCurve.shape)
        if approxCurve.shape[0] > 10:#用来区分圆6-12，矩形4，三角形3
            cv.drawContours(dst, contours, i, (0, 255, 0), 2)
    cv.imshow("detect contours", dst)

src = cv.imread("image1/009.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

measure_object(src)

cv.waitKey(0)
cv.destroyAllWindows()
