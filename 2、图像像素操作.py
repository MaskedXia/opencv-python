import cv2 as cv
import numpy as np
#图像操作
def access_pixels(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    print("height : %s, width : %s,channels : %s"%(height, width, channels))
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv = image[row, col, c]
                image[row, col, c] = 255 - pv
    cv.imshow("pixels", image)

def inverse(image):
    dst = cv.bitwise_not(image)
    cv.imshow("inversed", dst)

def create_image():
    img = np.zeros([400, 400, 3], np.uint8)
    img[: , : , 0] = np.ones([400, 400])*255
    cv.imshow("create1", img)

def pixel_change(image):
    print(image.shape)
    #print(image[140,140])#返回当前位置的BGR值
    # b, g, r = cv.split(image)#拆分及合并图像通道
    # image = cv.merge(b, g, r)
    copy = image.copy()
    copy[70:90,140:160] = [255,0,0]#也许使用 Numpy 的 array.item() 和 array.itemset() 会更好。但是返回值是标量
    cv.imshow("copy",copy)

src = cv.imread("image/1.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

t1 = cv.getTickCount()
#access_pixels(src)
t2 = cv.getTickCount()
time = (t2 - t1)/cv.getTickFrequency()#记录处理所花时间
print("time : %s ms"%(time * 1000))#ms单位

#create_image()
#inverse(src)
pixel_change(src)



cv.waitKey(0)
cv.destroyAllWindows()