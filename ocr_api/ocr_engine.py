import cv2
import numpy as np
from ocr_api.ctpn import CtpnDetector
from ocr_api.preprocessing import PreProcess
from ocr_api.tesseract_engine import TesseractEngine


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
        cv2.imwrite("out.png", processed_image)
        img = cv2.imread("out.png")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Deskew
        p = PreProcess()
        deskewed, angle = p.deskew(gray)
        cv2.imwrite("out.png", deskewed)

        # Text recognition using tesseract
        output_recognition = self.recogniser.img2txt(deskewed, 'eng')

        return output_recognition

