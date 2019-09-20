""" GMG.
Description : Main code that calls all the APIs
Author: Achraf KHAZRI Ai Reasearch Engineer
Project: GMG
"""

import sys
import argparse
import os
import glob


# from ocr_api.ocr_engine import OcrEngine
from speech_api.speech_engine import SpeechEngine


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--t', '-task', help="Classifier mode : task or train", type= str)
parser.add_argument('--m', '-method', help="Face detection method", type= str)
parser.add_argument('--i', '-input', help="input text object", type= str)

# Get args
args = parser.parse_args()

# Object instance
se = SpeechEngine()

# Face detection
if args.t == "face_detection":
    from face_api.face_engine import OpenCVHaarFaceDetector
    # Initialize method
    if args.m == 'opencv_haar':
        from face_api.face_engine import OpenCVHaarFaceDetector
        face_detector = OpenCVHaarFaceDetector()
        faces = face_detector.cascade_classifier_detector(args.i)

    elif args.m == 'dlib_hog':
        from face_api.face_engine import DlibHOGFaceDetector
        face_detector = DlibHOGFaceDetector()
        faces = face_detector.face_detector(args.i)

    elif args.m == 'dlib_cnn':
        from face_api.face_engine import DlibCNNFaceDetector
        face_detector = DlibCNNFaceDetector()
        faces = face_detector.detect_face(args.i)

    elif args.m == 'mtcnn':
        from face_api.face_engine import TensorflowMTCNNFaceDetector
        face_detector = TensorflowMTCNNFaceDetector()
        faces = face_detector.detect_face(args.i)

    elif args.m == 'mobilenet_ssd':
        from face_api.face_engine import TensoflowMobilNetSSDFaceDector
        face_detector = TensoflowMobilNetSSDFaceDector()
        faces = face_detector.detect_face(args.i)

    else:
        print("Error detection method !")

    print("Number of detected people is : " + str(len(faces)))
    se.text2speech("Number of detected people : " + str(len(faces)))


# Face recognition
elif args.t == "face_recognition":
    from face_api.face_engine import DlibHOGFaceDetector
    face_recogniser = DlibHOGFaceDetector()
    person = face_recogniser.dlib_recognition(args.i)
    print("Person : " + person)
    se.text2speech("We detected " + person)

# Initialise dataset to images in face_api/data/dataset
elif args.t == "face_init":
    from face_api.face_engine import DlibHOGFaceDetector
    face_recogniser = DlibHOGFaceDetector()
    person = face_recogniser.create_dataset()

# Add new person to dataset
elif args.t == "add_face":
    from face_api.face_engine import DlibHOGFaceDetector
    face_recogniser = DlibHOGFaceDetector()
    face_recogniser.add_face(args.i)

# Ask about somthing using wikipedia
elif args.t == "wiki":
    from wiki_api.wiki_engine import WikiEngine
    we = WikiEngine()
    infos = we.run(args.i, 3)
    print(infos)
    se.text2speech(infos)

else:
    print("Error command !")

