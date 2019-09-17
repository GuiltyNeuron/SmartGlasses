""" GMG.
Description : Main code that calls all the APIs
Author: Achraf KHAZRI Ai Reasearch Engineer
Project: GMG
"""

import sys
import argparse
import os
import glob
from face_api.face_engine import *

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--t', '-task', help="Classifier mode : task or train", type= str)
parser.add_argument('--p', '-path', help="File path", type= str)
parser.add_argument('--m', '-method', help="Face detection method", type= str)

# Get args
args = parser.parse_args()

# Object instance
fe = DlibHOGFaceDetector()
haar = OpenCVHaarFaceDetector()
# Face detection
if args.t == "face_detection":

    # Initialize method
    if args.m == 'opencv_haar':
        face_detector = OpenCVHaarFaceDetector()
        faces = face_detector.cascade_classifier_detector(args.p)

    elif args.m == 'dlib_hog':
        face_detector = DlibHOGFaceDetector()
        faces = face_detector.face_detector(args.p)

    elif args.m == 'dlib_cnn':
        face_detector = DlibCNNFaceDetector()
        faces = face_detector.detect_face(args.p)

    elif args.m == 'mtcnn':
        face_detector = TensorflowMTCNNFaceDetector()
        faces = face_detector.detect_face(args.p)

    elif args.m == 'mobilenet_ssd':
        face_detector = TensoflowMobilNetSSDFaceDector()
        faces = face_detector.detect_face(args.p)

    else:
        print("Error detection method !")

    print("Number of detected people : " + str(len(faces)))


# Face recognition
elif args.t == "face_recognition":

    face_recogniser = DlibHOGFaceDetector()
    person = face_recogniser.dlib_recognition(args.p)
    print("Person : " + person)

# Initialise dataset to images in face_api/data/dataset
elif args.t == "face_init":
    face_recogniser = DlibHOGFaceDetector()
    person = face_recogniser.create_dataset()

# Add new person to dataset
elif args.t == "add_face":
    face_recogniser = DlibHOGFaceDetector()
    face_recogniser.add_face(args.p)

else:
    print("Error command !")