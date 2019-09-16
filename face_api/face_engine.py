import cv2 as cv
import face_recognition
import os
import pandas as pd
import numpy as np
import pickle


class faceEngine():

    def cascade_classifier_detector(self, img):
        """
        Face detection using Haar cascade classifier with OpenCv
        :param img: input image
        :return: faces bboxes
        """

        # Load Haar Cascade Classifier
        face_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')
        eye_cascade = cv.CascadeClassifier('data/haarcascade_eye.xml')

        # Convert image to graysclae
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Detec faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
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

    def dlib_detector(self, img):
        """
        Face detection using Dlib
        :param img: input image
        :return: faces bboxes
        """

        # Find all the faces in the image
        face_locations = face_recognition.face_locations(img)

        # Get number of faces
        """number_of_faces = len(face_locations)
        print("Number of face(s) in this image {}.".format(number_of_faces))

        for face_location in face_locations:
            # Get coord
            x, y, z, w = face_location

            # Draw Face rectangle
            cv.rectangle(img, (w, x), (y, z), (0, 0, 255), 2)

        # Show image
        cv.imshow("img", img)
        cv.imwrite("detected.jpg", img)
        cv.waitKey(0)
        cv.destroyAllWindows()"""

        return face_locations

    def dlib_recognition(self, img_path):
        """
        Face recognition using Dlib
        :param img: input image
        :return: Names of the detected persons
        """
        output_name = "No_Body"

        # Load dataset
        dataset = pd.read_pickle("data/data.pkl")

        encodings = dataset["encodings"].values
        names = dataset["names"].values
        # Load image
        img = face_recognition.load_image_file(img_path)

        # Encode facial features
        encoding = face_recognition.face_encodings(img)[0]
        i = 0

        for encode in encodings:

            result = face_recognition.compare_faces([encode], encoding)

            if result[0] == True:
                output_name = names[i]
                break

            i += 1

        return output_name

    def create_dataset(self, folder_path):

        # Faces encoding features
        features = []

        # Persons names extracted from images names
        names = []

        # Filter all files in the directory
        for filename in os.listdir(folder_path):

            # Make sure that our file is text
            if (filename.endswith('.jpeg')) or (filename.endswith('.jpg')) or (filename.endswith('.png')):

                # Load image
                image = face_recognition.load_image_file(folder_path + "/" + filename)

                # Encode face features
                feature = face_recognition.face_encodings(image)[0]

                # Appen feature to total features list
                features.append(feature)

                # Add person name to dataset
                names.append(os.path.splitext(filename)[0])

                print("Encoded <<<>>>" + os.path.splitext(filename)[0] + "<<<>>>")

        # Save dataset
        data = pd.DataFrame({"encodings": features, "names": names})
        pd.to_pickle(data, "data/data.pkl")

    def add_face(self, image_path):

        # Load dataset
        dataset = pd.read_pickle("data/data.pkl")

        # Get old data
        names = list(dataset['names'].values)
        features = list(dataset['encodings'].values)

        # Get file name
        filename = os.path.basename(image_path)

        # Add name to other names
        names.append(os.path.splitext(filename)[0])

        print(names)
        # Load image
        image = face_recognition.load_image_file(image_path)

        # Encode face features
        feature = face_recognition.face_encodings(image)[0]

        # Append feature to total features list
        features.append(feature)

        # Save dataset
        data = pd.DataFrame({"encodings": features, "names": names})
        pd.to_pickle(data, "data/data.pkl")

fd = faceEngine()

# Folder that contains all the images
folder_path = "data/dataset"

#fd.create_dataset(folder_path)

#fd.add_face("data/Albert_Einstein.jpg")
out = fd.dlib_recognition("data/Albert_Einstein.jpg")
print(out)