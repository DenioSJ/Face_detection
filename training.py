import numpy as np
import cv2
import os

import face_recognition as fr

test_img=cv2.imread(r'C:\Users\denio\Desktop\Courses & Certificates\courses\DataScience\face recog own\test.jpeg')

faces_detected,gray_img=fr.faceDetection(test_img)
print("Face Detected: ",faces_detected)

faces,face_id=fr.labels_for_training_data(r'C:\Users\denio\Desktop\Courses & Certificates\courses\DataScience\face recog own\images')

face_recognizer=fr.train_Classifier(faces,face_id)

face_recognizer.save(r'C:\Users\denio\Desktop\Courses & Certificates\courses\DataScience\face recog own\trainingdata.yml')

name={0:'Denio'}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+w,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print(label)
    print(confidence)
    fr.draw_rectangle(test_img,face)
    predict_name=name[label]
    fr.put_text(test_img,predict_name,x,y)

resized_img=cv2.resize(test_img,(1000,700))

cv2.imshow('Resized',resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()