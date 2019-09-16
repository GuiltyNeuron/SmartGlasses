""" GMG.
Description : Main code that calls all the APIs
Author: Achraf KHAZRI Ai Reasearch Engineer
Project: GMG
"""

import sys
import argparse
import os
import glob
from face_api.face_engine import faceEngine

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--t', '-task', help="Classifier mode : task or train", type= str)
parser.add_argument('--p', '-path', help="File path", type= str)

# Get args
args = parser.parse_args()

# Object instance
fe = faceEngine()

# Face detection
if args.t == "face_detection":
    faces = fe.dlib_detector(args.p)
    print("Number of detected people : " + str(len(faces)))

# Face recognition
elif args.t == "face_recognition":
    person = fe.dlib_recognition(args.p)
    print("Person : " + person)

# Initialise dataset to images in face_api/data/dataset
elif args.t == "face_init":
    person = fe.create_dataset()
    print("Done !")

# Add new person to dataset
elif args.t == "add_face":
    fe.add_face(args.p)

else:
    print("Error command !")