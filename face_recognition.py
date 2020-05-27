import numpy as np
import cv2
import os


def faceDetection(input_img):
    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    face_haar = cv2.CascadeClassifier(
        r"C:\Users\denio\Desktop\Courses & Certificates\courses\DataScience\face recog own\haarcascade_frontalface_alt.xml"
    )
    faces = face_haar.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=3)
    return faces, gray_img


def labels_for_training_data(directory):
    faces = []
    faceid = []

    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file")
                continue
            id = os.path.basename(path)
            img_path = os.path.join(path, filename)
            print("img_path", img_path)
            print("id", id)
            test_img = cv2.imread(img_path)
            if test_img is None:
                print("Not Loaded Properly")
                continue

            faces_rect, gray_img = faceDetection(test_img)
            (x, y, w, h) = faces_rect[0]
            roi_gray = gray_img[y : y + w, x : x + h]
            faces.append(roi_gray)
            faceid.append(int(id))
    return faces, faceid


def train_Classifier(faces, faceid):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceid))
    return face_recognizer


def draw_rectangle(test_img, face):
    (x, y, w, h) = face
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=3)


def put_text(test_img, label_name, x, y):
    cv2.putText(
        test_img, label_name, (x, y), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 0), 3
    )
