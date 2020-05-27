import cv2
import sys

count = 0

vidStream = cv2.VideoCapture(0)

while True:
    ret, frame = vidStream.read()
    cv2.imshow("Test", frame)

    cv2.imwrite(
        r"C:\Users\denio\Desktop\Courses & Certificates\courses\DataScience\face recog own\images\0\image%04i.jpg"
        % count,
        frame,
    )
    count = count + 1

    if cv2.waitKey(10) == ord("q"):
        break
