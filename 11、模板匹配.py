import cv2 as cv
import numpy as np
#模板匹配
def template_demo():
    tpl = cv.imread("image1/005.png")
    target = cv.imread("image/5.png")
    #cv.imshow("template image", tpl)
    #cv.imshow("target image", target)
    methods = [cv.TM_SQDIFF_NORMED,  cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]
    #CV.TM_SQDIFF 平方差匹配法, 最好的匹配值为0
    # CV.TM_CCORR 相关匹配法：该方法采用乘法操作,数值越大表明匹配程度越好
    #CV_TM_CCOEFF 相关系数匹配法,1表示完美的匹配；-1表示最差的匹配
    th, tw = tpl.shape[:2]
    for md in methods:
        print(md)
        result = cv.matchTemplate(target, tpl, md)
        #用模板图像在输入图像（大图）上滑动，并在每一个位置对模板图像和与其对应的
        # 输入图像的子区域进行比较
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        #返回矩阵result的最小值、最大值、最小值索引、最大值索引
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc#越小越好，得到最小值索引
        else:
            tl = max_loc
        br = (tl[0]+tw, tl[1]+th)
        cv.rectangle(target, tl, br, (0,0,255), 2)
        cv.imshow("match"+np.str(md), result)
        cv.imshow("after_match", target)

src = cv.imread("image/1.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

template_demo()

cv.waitKey(0)
cv.destroyAllWindows()