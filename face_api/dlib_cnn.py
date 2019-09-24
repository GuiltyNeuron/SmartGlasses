import cv2 as cv
import dlib
import numpy as np


class DlibCNNFaceDetector():
    def __init__(self, nrof_upsample=0, model_path=os.path.join(os.path.dirname(__file__), 'models/mmod_human_face_detector.dat')):
        
        self.cnn_detector = dlib.cnn_face_detection_model_v1(model_path)
        self.nrof_upsample = nrof_upsample

    def detect_face(self, image_path):
        # Load image
        image = cv.imread(image_path)

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