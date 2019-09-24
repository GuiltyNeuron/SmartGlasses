import cv2 as cv
import os

class OpenCVHaarFaceDetector():

    def __init__(self):
        # Load Haar Cascade Classifier
        self.path = os.path.join(os.path.dirname(__file__), 'data/haarcascade_frontalface_default.xml')
        self.face_cascade = cv.CascadeClassifier(self.path)

    def cascade_classifier_detector(self, img_path):
        """
        Face detection using Haar cascade classifier with OpenCv
        :param img: input image
        :return: faces bboxes
        """

        # Load image
        img = cv.imread(img_path)

        # Convert image to graysclae
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Detec faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        """for (x, y, w, h) in faces:
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = img[y:y + h, x:x + w]
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    for (ex, ey, ew, eh) in eyes:
                        cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                cv.imwrite("detection.jpg", img)
                cv.imshow('img', img)
                cv.waitKey(0)"""
        return faces