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
parser.add_argument('--l', '-language', help="output language", type= str)
parser.add_argument('--city', '-city', help="city name", type= str)
parser.add_argument('--country', '-country', help="country name", type= str)



# Get args
args = parser.parse_args()

# Object instance
se = SpeechEngine(args.l)

# Face detection
if args.t == "face_detection":

    if args.m == 'opencv_haar':
        from face_api.haar_cascade import OpenCVHaarFaceDetector
        face_detector = OpenCVHaarFaceDetector()
        faces = face_detector.cascade_classifier_detector(args.i)

    elif args.m == 'dlib_hog':
        from face_api.dlib_hog import DlibHOGFaceDetector
        face_detector = DlibHOGFaceDetector()
        faces = face_detector.face_detector(args.i)

    elif args.m == 'dlib_cnn':
        from face_api.dlib_cnn import DlibCNNFaceDetector
        face_detector = DlibCNNFaceDetector()
        faces = face_detector.detect_face(args.i)

    elif args.m == 'mtcnn':
        from face_api.mtcnn import TensorflowMTCNNFaceDetector
        face_detector = TensorflowMTCNNFaceDetector()
        faces = face_detector.detect_face(args.i)

    elif args.m == 'mobilenet_ssd':
        from face_api.ssd_mobilenet import TensoflowMobilNetSSDFaceDector
        face_detector = TensoflowMobilNetSSDFaceDector()
        faces = face_detector.detect_face(args.i)

    else:
        print("Error detection method !")

    if args.l == "en":
        if len(faces) == 0:
            se.text2speech("we detected no body !")

        elif len(faces) == 1:
            se.text2speech("One person was detected !")
        else:
            se.text2speech(str(len(faces)) + " persons were detected !")

    elif args.l == "fr":
        if len(faces) == 0:
            se.text2speech("Aucune personne a été détecté ! ")

        elif len(faces) == 1:
            se.text2speech("Une personne a été détecté !")
        else:
            se.text2speech(str(len(faces)) + " personnes ont été détectés ! ")


# Face recognition
elif args.t == "face_recognition":
    from face_api.dlib_hog import DlibHOGFaceDetector
    face_recogniser = DlibHOGFaceDetector()
    person = face_recogniser.dlib_recognition(args.i)

    if args.l == "en":
        if person == "no body !":
            se.text2speech("We couldn't recognise any person !")
        else:
            se.text2speech(person + " was recognised !")

    elif args.l == "fr":
        if person == "no body !":
            se.text2speech("Aucunne personne a été reconnu !")
        else:
            se.text2speech(person + " a été reconnu !")

    else:
        se.text2speech("Error language !")

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
    from infos_api.wiki_engine import WikiEngine
    if args.l == "fr":
        we = WikiEngine("fr")
        infos = we.run(args.i, 3)
        se.text2speech(infos)
    elif args.l == "en":
        we = WikiEngine("en")
        infos = we.run(args.i, 3)
        se.text2speech(infos)

    else:
        print("Error selectiong language !")

elif args.t == "news_latest":
    from infos_api.news_engine import NewsEngine

    news = NewsEngine()
    articles, links = news.get_latest_articles()
    se.text2speech("We are about to read the latest news articles posted by CNN.")
    i = 1
    for article in articles:
        output = "Article number " + str(i) + ". " + article
        se.text2speech(output)
        i +=1

elif args.t == "news_article":
    from infos_api.news_engine import NewsEngine

    article_number = int(args.i)
    news = NewsEngine()
    articles, links = news.get_latest_articles()
    article_txt = news.get_article(article_number)
    article_title = articles[article_number]
    se.text2speech("We are about to read the article titeled : " + articles[article_number - 1])
    se.text2speech(article_txt)
    se.text2speech("I hope you enjoyed the article from CNN news.")

elif args.t == "weather":
    from infos_api.weather_engine import WeatherEngine

    we = WeatherEngine(args.country, args.city)
    weather = we.get_today_weather()
    se.text2speech(weather)

elif args.t == "date":
    from infos_api.time_engine import TimeEngine
    te = TimeEngine()
    date = te.date()
    se.text2speech(date)

elif args.t == "time":
    from infos_api.time_engine import TimeEngine
    te = TimeEngine()
    time = te.time()
    se.text2speech(time)

elif args.t == "ocr":
    from ocr_api.ocr_engine import OcrEngine

    ocr = OcrEngine()
    txt = ocr.run(args.i)
    se.text2speech(txt)
else:
    print("Error command !")


