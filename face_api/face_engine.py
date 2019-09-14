import cv2 as cv
import face_recognition


class faceDetector():

    def cascade_classifier_detector(self, img):

        # Load Haar Cascade Classifier
        face_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')
        eye_cascade = cv.CascadeClassifier('data/haarcascade_eye.xml')

        # Convert image to graysclae
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Detec faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv.imwrite("detection.jpg", img)
        cv.imshow('img', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def dlib_detector(self, img):

        # Find all the faces in the image
        face_locations = face_recognition.face_locations(img)

        # Get number of faces
        number_of_faces = len(face_locations)
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
        cv.destroyAllWindows()


# Load the jpg file into a numpy array
image = cv.imread("data/test.jpg")

fd = faceDetector()
fd.dlib_detector(image)