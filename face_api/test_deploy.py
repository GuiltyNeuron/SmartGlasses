import argparse
import cv2
from face_detector import *


def get_args():
    parser = argparse.ArgumentParser(
        description=
        "This example script will show you how to use the face detector module.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to image file")
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default='dlib_hog',
        help=
        "Please choose between : opencv_haar , dlib_hog , dlib_cnn , mtcnn , mobilenet_ssd "
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Read image file using opencv
    image_data = cv2.imread(args.input)

    # Current available method in this repo
    method_list = [
        'opencv_haar', 'dlib_hog', 'dlib_cnn', 'mtcnn', 'mobilenet_ssd'
    ]
    method = args.method

    # Initialize method
    if method == 'opencv_haar':
        face_detector = OpenCVHaarFaceDetector(
            scaleFactor=1.3,
            minNeighbors=5,
            model_path='models/haarcascade_frontalface_default.xml')

    elif method == 'dlib_hog':
        face_detector = DlibHOGFaceDetector(
            nrof_upsample=0, det_threshold=-0.2)

    elif method == 'dlib_cnn':
        face_detector = DlibCNNFaceDetector(
            nrof_upsample=0, model_path='models/mmod_human_face_detector.dat')

    elif method == 'mtcnn':
        face_detector = TensorflowMTCNNFaceDetector(model_path='models/mtcnn')

    elif method == 'mobilenet_ssd':
        face_detector = TensoflowMobilNetSSDFaceDector(
            det_threshold=0.3,
            model_path='models/ssd/frozen_inference_graph_face.pb')

    if method not in method_list:
        print 'Please select the available method from this list: opencv_haar , dlib_hog , dlib_cnn , mtcnn , mobilenet_ssd'
    else:

        # Detect face in the image
        detected_face = face_detector.detect_face(image_data)

        # Resulting detection will be in numpy array format
        # E.g :
        #
        # [[xmin,ymin,xmax,ymax]
        # 			...
        #  [xmin,ymin,xmax,ymax]]

        # Visualizing detected face
        for face in detected_face:

            # Usage : cv2.rectangle( image_data,(xmin,ymin),(xmax,ymax),colour in BGR format : e.g (255,0,0) , line thickness = e.g 2 )
            cv2.rectangle(image_data, (face[0], face[1]), (face[2], face[3]),
                          (255, 0, 0), 2)

        # Show result
        cv2.imshow('Face Detection', image_data)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
