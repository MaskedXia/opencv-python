import cv2 as cv
import numpy as np
#读取视频
def video_demo():
    capture = cv.VideoCapture("video/01.mp4")#内置设备0的摄像头，也可以是个视频文件
    # print(capture.isOpened())#初始化摄像头设备,否则就要使用函数 cap.open()
    # cap.get(3)和cap.get(4)来查看每一帧的宽和高。默认情况下得到的值是640X480。
    # 但是我可以使用ret = cap.set(3, 320)和ret = cap.set(4, 240)来把宽和高改成
    # 320X240
    while(True):
        ret, frame = capture.read()#frame-by-frame的图像
        #frame = cv.flip(frame, 1) #垂直方向旋转
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow("video", frame)
        c = cv.waitKey(50)# cv2.waiKey() 设置适当的持续时间
        if c == 27:
            break
    capture.release()
    # 保存视频
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    # out.write(frame)
    # 已经装了合适版本的ffmpeg或者gstreamer



def get_image_info(image):
    print(type(image))#nd.array
    print(image.shape)#[,,]
    print(image.size)#像素数目
    print(image.dtype)#图像数据类型uint8
    pixel_data = np.array(image)
    print(pixel_data)


src = cv.imread("image/1.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
# get_image_info(src)
# gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# cv.imwrite("image1/02.png", gray)

video_demo()

cv.waitKey(0)
cv.destroyAllWindows()