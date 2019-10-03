import face_api.resources.mtcnn.mtcnn as mtcnn
import cv2 as cv
import os
import tensorflow as tf

class TensorflowMTCNNFaceDetector():
    def __init__(self, model_path=os.path.join(os.path.dirname(__file__), 'models/mtcnn')):
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

    def detect_face(self, image_path):
        # Load image
        image = cv.imread(image_path)

        dets, face_landmarks = mtcnn.detect_face(
            image, self.minsize, self.pnet, self.rnet, self.onet,
            self.threshold, self.factor)

        faces = dets[:, :4].astype('int')
        conf_score = dets[:, 4]

        return faces