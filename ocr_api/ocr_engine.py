from __future__ import print_function
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import _get_blobs
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
from lib.rpn_msr.proposal_layer_tf import proposal_layer
from tensorflow.python.platform import gfile
import pytesseract
import math
import glob
import os
import shutil
import sys
import cv2
import numpy as np
import tensorflow as tf


class CtpnDetector:

    def __init__(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        with gfile.FastGFile('data/ctpn.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        self.sess.run(tf.global_variables_initializer())

        self.input_img = self.sess.graph.get_tensor_by_name('Placeholder:0')
        self.output_cls_prob = self.sess.graph.get_tensor_by_name('Reshape_2:0')
        self.output_box_pred = self.sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')

    def resize_im(self, im, scale, max_scale=None):
        f = float(scale) / min(im.shape[0], im.shape[1])
        if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
            f = float(max_scale) / max(im.shape[0], im.shape[1])
        return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f

    def detect_text(self, img):

        # resize image
        img, scale = self.resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)

        blobs, im_scales = _get_blobs(img, None)
        if cfg.TEST.HAS_RPN:
            im_blob = blobs['data']
            blobs['im_info'] = np.array(
                [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
                dtype=np.float32)
        cls_prob, box_pred = self.sess.run([self.output_cls_prob, self.output_box_pred], feed_dict={self.input_img: blobs['data']})
        rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)

        scores = rois[:, 0]
        boxes = rois[:, 1:5] / im_scales[0]
        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        output = []
        for box in boxes:
            min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))

            output.append([min_y, max_y, min_x, max_x])
        return output, boxes

    def draw_boxes(self, img, image_name, boxes, scale):
        base_name = image_name.split('\\')[-1]
        with open('data\\results\\' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
            for box in boxes:
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                if box[8] >= 0.9:
                    color = (0, 255, 0)
                elif box[8] >= 0.8:
                    color = (255, 0, 0)
                cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
                cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
                cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

                min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
                min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
                max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
                max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))

                line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
                f.write(line)

class TesseractEngine:

    def __init__(self):
        self.tessdata_dir_config = r'--tessdata-dir "tessdata"'
        self.tesseract_exec_path = r'C:\tesseract\tesseract\bin\tesseract.exe'
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_exec_path

    def img2txt(self, img, language):
        return pytesseract.image_to_string(img, lang = language, config=self.tessdata_dir_config)


class PreProcess():

    def deskew(self, img):

        gray = cv2.bitwise_not(img)

        # threshold the image, setting all foreground pixels to
        # 255 and all background pixels to 0
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # grab the (x, y) coordinates of all pixel values that
        # are greater than zero, then use these coordinates to
        # compute a rotated bounding box that contains all
        # coordinates

        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]

        # the `cv2.minAreaRect` function returns values in the
        # range [-90, 0); as the rectangle rotates clockwise the
        # returned angle trends to 0 -- in this special case we
        # need to add 90 degrees to the angle
        if angle < -45:
            angle = -(90 + angle)

        # otherwise, just take the inverse of the angle to make
        # it positive
        else:
            angle = -angle

        # rotate the image to deskew it
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated, angle


class OcrEngine():

    def __init__(self):
        self.detector = CtpnDetector()
        self.recogniser = TesseractEngine()

    def run(self, image_path):
        # Load image
        img = cv2.imread(image_path)

        # BGR to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold

        # blur = cv2.GaussianBlur(gray, (5, 5), 0)

        th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        mask = np.zeros(shape=th.shape)

        # Detect text
        output, bboxes = self.detector.detect_text(img)

        for i in range(len(output)):

            # mini mask for every crop
            m_ones = np.ones(shape=(output[i][1] - output[i][0], output[i][3] - output[i][2]))
            m = np.pad(m_ones, ((output[i][0], gray.shape[0] - output[i][1]), (output[i][2], gray.shape[1] - output[i][3])), 'constant')

            # add mini mask to global mask
            mask = mask + m

        # Delete unneaded area
        image_text_only = th * mask

        # Add white to unneaded area
        white = np.zeros(shape=th.shape)
        white_mask = np.where(mask == 0, 255, white)
        processed_image = image_text_only + white_mask

        # Save and load processed image
        cv2.imwrite("data/demo/out.png", processed_image)
        img = cv2.imread("data/demo/out.png")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Deskew
        p = PreProcess()
        deskewed, angle = p.deskew(gray)
        cv2.imwrite("data/demo/out.png", deskewed)

        # Text recognition using tesseract
        output_recognition = self.recogniser.img2txt(deskewed, 'eng')

        print(output_recognition)


e = OcrEngine()
e.run("data/demo/deskew2.jpg")

