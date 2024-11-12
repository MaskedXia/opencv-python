import cv2 as cv#人脸检测 Haar， LBP
import numpy as np

def face_detect_demo(src):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier(r"opencv-master\data\haarcascades\haarcascade_frontalface_alt_tree.xml")
    faces = face_detector.detectMultiScale(gray, 1.02, 5)
    for x, y, w, h in faces:
        cv.rectangle(src, (x, y), (x+w, y+h), (0,0,255),2)
    cv.imshow("result", src)
    cv.waitKey(0)


#src = cv.imread("image/3.png")
capture = cv.VideoCapture(0)
#cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.namedWindow("result", cv.WINDOW_AUTOSIZE)
#cv.imshow("input image", src)
while(True):
    ret, frame = capture.read()
    frame = cv.flip(frame, 1)
    face_detect_demo(frame)
    c = cv.waitKey(10)
    if c == 27:
        break
#face_detect_demo()

cv.waitKey(0)
cv.destroyAllWindows()