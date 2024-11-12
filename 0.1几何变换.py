import cv2 as cv
import numpy as np

#cv.resize  插值方法
'''
INTER_NEAREST：最近邻插值   （四邻象素中，将距离待求象素最近的邻象素灰度赋给待求象素）
INTER_LINEAR：双线性插值（默认设置）    扩展时推荐
INTER_AREA：使用像素区域关系进行重采样    缩放时推荐
INTER_CUBIC：4x4像素邻域的双三次插值（慢）  扩展时推荐
INTER_LANCZOS4：8x8像素邻域的Lanczos插值
'''
def expand_shrink(image):#扩展和缩放
    res = cv.resize(image, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    #上面的 None 本应该是输出图像的尺寸，但是因为后边我们设置了缩放因子
    # height, width = img.shape[:2]  #或者这样
    # res = cv2.resize(img, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)
    cv.imshow("res", res)
    print(res.shape)

def translation(image):#平移和旋转
    height, width = image.shape[:2]
    M = cv.getRotationMatrix2D((width/2, height/2), 45, 0.5)
    # 旋转的中心点（center），旋转角度()，旋转后缩放因子

    # M = np.float32([[1,0,100],[0,1,50]])
    # tx = 100, ty = 50, 即水平平移100，垂直平移50

    # pts1 = np.float32([[50, 50], [100, 50], [50, 200]])
    # pts2 = np.float32([[10, 100], [100, 50], [100, 250]])
    # M = cv.getAffineTransform(pts1, pts2)
    #仿射

    # pts1 = np.float32([[56, 65], [200, 52], [28, 200], [200, 200]])
    # pts2 = np.float32([[0, 0], [100, 0], [0, 100], [100, 100]])
    # M = cv.getPerspectiveTransform(pts1, pts2)
    # dst = cv.warpPerspective(image, M, (width, height))
    # 透视

    dst = cv.warpAffine(image, M, (width, height))
    print(dst.shape)
    cv.imshow("translation", dst)

#像素值高于阈值时，我们给这个像素赋予一个新值,否则我们给它赋予另外一种颜色
'''
cv2.THRESH_BINARY        大于取最大值
cv2.THRESH_BINARY_INV     小于取最大值
cv2.THRESH_TRUNC          大于取阈值，小于取原来值
cv2.THRESH_TOZERO          大于取原来值
cv2.THRESH_TOZERO_INV       小于取原来值
'''
def threshold(image):#全局阈值
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, dst1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    #ret, dst1 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    #自己寻找最优阈值
    cv.imshow("threshold", dst1)

def adaptive_threshold(image):#自适应阈值
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, dst1 = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
    dst2 = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                cv.THRESH_BINARY, 3, 2)
    #src, maxval,
    # cv2.ADPTIVE_THRESH_MEAN_C：阈值取自相邻区域的平均值
    # cv2.ADPTIVE_THRESH_GAUSSIAN_C：阈值取值相邻区域的加权和，权重为一个高斯窗口
    # Block Size - 邻域大小  越小图像越细致
    #C - 这就是是一个常数，阈值就等于的平均值或者加权平均值减去这个常数
    dst3 = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv.THRESH_BINARY, 3, 2)
    cv.imshow("DAPTIVE_THRESH_MEAN_C", dst2)
    cv.imshow("ADAPTIVE_THRESH_GAUSSIAN_C", dst3)


src = cv.imread("image/1.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
print(src.shape)

#expand_shrinkc
#translation(src)
#threshold(src)
adaptive_threshold(src)


cv.waitKey(0)
cv.destroyAllWindows()