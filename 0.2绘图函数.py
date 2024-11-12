import numpy as np
import cv2 as cv
#  img：你想要绘制图形的那幅图像。
# • color：形状的颜色。以 RGB 为例，需要传入一个元组，例如： （255,0,0 ）
# 代表蓝色。对于灰度图只需要传入灰度值。
# • thickness：线条的粗细。如果给一个闭合图形设置为 -1，那么这个图形
# 就会被填充。默认值是 1.
# • linetype：线条的类型，8 连接，抗锯齿等。默认情况是 8 连接。cv2.LINE_AA
# 为抗锯齿，这样看起来会非常平滑

#画线
img = np.zeros([512,512,3], np.uint8)
#cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.line(img, (0,0), (512,512), (0,0,255), 2)#起点和终点


#画矩形
cv.rectangle(img, (50,50), (150,150), (0,255,0), 2)#左上角顶点和右下角顶点


#画圆
cv.circle(img, (75, 75), 30, (255,0,0), 2)#中心点坐标和半径大小


#画椭圆
cv.ellipse(img, (256,256), (100,50), 30, 0, 180, (255,255,0), -1)
# 一个参数是中心点的位置坐标。
# 下一个参数是长轴和短轴的长度。椭圆沿逆时针方向旋转的角度。椭圆弧演
# 顺时针方向起始的角度和结束角度，如果是 0 很 360，就是整个椭圆


#添加文字
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img, "...", (50,500), font, 2, (0,255,255), 2)
#图像，文字，位置，类型，文字大小，颜色，粗细


#画边框
constant = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_CONSTANT,value=(255,255,0))
#图像， top, bottom, left, right 对应边界的像素数目，类型，颜色
cv.imshow("image", img)






cv.waitKey(0)
cv.destroyAllWindows()