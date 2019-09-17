import cv2 as cv
import face_recognition
import os
import sys
import pandas as pd
import numpy as np
import pickle


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


class DlibHOGFaceDetector():

    def __init__(self):

        # Relative dataset path to this module
        self.dataset_path = os.path.join(os.path.dirname(__file__), 'data/data.pkl')

        # Relative dataset images path to this module
        self.face_images_path = os.path.join(os.path.dirname(__file__), 'data/dataset')

    def face_detector(self, img_path):
        """
        Face detection using Dlib
        :param img: input image
        :return: faces bboxes
        """

        # Load image
        img = face_recognition.load_image_file(img_path)

        # Find all the faces in the image
        face_locations = face_recognition.face_locations(img)

        return face_locations

    def dlib_recognition(self, img_path):
        """
        Face recognition using Dlib
        :param img: input image
        :return: Names of the detected persons
        """
        output_name = "No_Body"

        # Load dataset
        dataset = pd.read_pickle(self.dataset_path)

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

    def create_dataset(self):

        # Faces encoding features
        features = []

        # Persons names extracted from images names
        names = []

        # Filter all files in the directory
        for filename in os.listdir(self.face_images_path):

            # Make sure that our file is text
            if (filename.endswith('.jpeg')) or (filename.endswith('.jpg')) or (filename.endswith('.png')):

                # Load image
                image = face_recognition.load_image_file(self.face_images_path + "/" + filename)

                # Encode face features
                feature = face_recognition.face_encodings(image)[0]

                # Appen feature to total features list
                features.append(feature)

                # Add person name to dataset
                names.append(os.path.splitext(filename)[0])

                print("<<<<<<<< Encoded >>>>>>>>>>> <<<<<<<< " + os.path.splitext(filename)[0] + " >>>>>>>>>>>")

        # Save dataset
        data = pd.DataFrame({"encodings": features, "names": names})
        pd.to_pickle(data, self.dataset_path)

    def add_face(self, image_path):

        # Load dataset
        dataset = pd.read_pickle(self.dataset_path)

        # Get old data
        names = list(dataset['names'].values)
        features = list(dataset['encodings'].values)

        # Get file name
        filename = os.path.basename(image_path)

        # Person name
        name = os.path.splitext(filename)[0]

        # Check if the person already exists
        exist = False
        for n in names:
            if n == name:
                exist = True
                break

        if(exist == False):
            # Add name to other names
            names.append(name)

            # Load image
            image = face_recognition.load_image_file(image_path)

            # Encode face features
            feature = face_recognition.face_encodings(image)[0]

            # Append feature to total features list
            features.append(feature)

            # Save dataset
            data = pd.DataFrame({"encodings": features, "names": names})
            pd.to_pickle(data, self.dataset_path)

            print(os.path.splitext(filename)[0] + " Added successefuly !")
        else:
            print(name + " already exist !")


class DlibCNNFaceDetector():
    def __init__(self,
                 nrof_upsample=0,
                 model_path='models/mmod_human_face_detector.dat'):

        self.cnn_detector = dlib.cnn_face_detection_model_v1(model_path)
        self.nrof_upsample = nrof_upsample

    def detect_face(self, image):

        dets = self.cnn_detector(image, self.nrof_upsample)

        faces = []
        for i, d in enumerate(dets):
            x1 = int(d.rect.left())
            y1 = int(d.rect.top())
            x2 = int(d.rect.right())
            y2 = int(d.rect.bottom())
            score = float(d.confidence)

            faces.append(np.array([x1, y1, x2, y2]))

        return np.array(faces)
    

class TensorflowMTCNNFaceDetector():
    def __init__(self, model_path='models/mtcnn'):

        self.minsize = 15
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor

        with tf.Graph().as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = mtcnn.create_mtcnn(
                    self.sess, model_path)

    def detect_face(self, image):

        dets, face_landmarks = mtcnn.detect_face(
            image, self.minsize, self.pnet, self.rnet, self.onet,
            self.threshold, self.factor)

        faces = dets[:, :4].astype('int')
        conf_score = dets[:, 4]

        return faces


class TensoflowMobilNetSSDFaceDector():
    def __init__(self,
                 det_threshold=0.3,
                 model_path='models/ssd/frozen_inference_graph_face.pb'):

        self.det_threshold = det_threshold
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.detection_graph, config=config)

    def detect_face(self, image):

        h, w, c = image.shape

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')

        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name(
            'num_detections:0')

        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)

        filtered_score_index = np.argwhere(
            scores >= self.det_threshold).flatten()
        selected_boxes = boxes[filtered_score_index]

        faces = np.array([[
            int(x1 * w),
            int(y1 * h),
            int(x2 * w),
            int(y2 * h),
        ] for y1, x1, y2, x2 in selected_boxes])

        return faces
