import cv2
import numpy as np
from face_detector import *
import time
import os
from tqdm import tqdm
import time
import argparse


def get_iou(boxA, boxB):
    """
	Calculate the Intersection over Union (IoU) of two bounding boxes.

	Parameters
	----------
	boxA = np.array( [ xmin,ymin,xmax,ymax ] )
	boxB = np.array( [ xmin,ymin,xmax,ymax ] )

	Returns
	-------
	float
		in [0, 1]
	"""

    bb1 = dict()
    bb1['x1'] = boxA[0]
    bb1['y1'] = boxA[1]
    bb1['x2'] = boxA[2]
    bb1['y2'] = boxA[3]

    bb2 = dict()
    bb2['x1'] = boxB[0]
    bb2['y1'] = boxB[1]
    bb2['x2'] = boxB[2]
    bb2['y2'] = boxB[3]

    # Determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes area
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    assert iou >= 0.0
    assert iou <= 1.0

    return iou


def extract_and_filter_data(splits):
    # Extract bounding box ground truth from dataset annotations, also obtain each image path
    # and maintain all information in one dictionary
    bb_gt_collection = dict()

    for split in splits:
        with open(
                os.path.join('dataset', 'wider_face_split',
                             'wider_face_%s_bbx_gt.txt' % (split)), 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.split('\n')[0]
            if line.endswith('.jpg'):
                image_path = os.path.join('dataset', 'WIDER_%s' % (split),
                                          'images', line)
                bb_gt_collection[image_path] = []
            line_components = line.split(' ')
            if len(line_components) > 1:

                # Discard annotation with invalid image information, see dataset/wider_face_split/readme.txt for details
                if int(line_components[7]) != 1:
                    x1 = int(line_components[0])
                    y1 = int(line_components[1])
                    w = int(line_components[2])
                    h = int(line_components[3])

                    # In order to make benchmarking more valid, we discard faces with width or height less than 15 pixel,
                    # we decide that face less than 15 pixel will not informative enough to be detected
                    if w > 15 and h > 15:
                        bb_gt_collection[image_path].append(
                            np.array([x1, y1, x1 + w, y1 + h]))

    return bb_gt_collection


def evaluate(face_detector, bb_gt_collection, iou_threshold):
    total_data = len(bb_gt_collection.keys())
    data_total_iou = 0
    data_total_precision = 0
    data_total_inference_time = 0

    # Evaluate face detector and iterate it over dataset
    for i, key in tqdm(enumerate(bb_gt_collection), total=total_data):
        image_data = cv2.imread(key)
        face_bbs_gt = np.array(bb_gt_collection[key])
        total_gt_face = len(face_bbs_gt)

        start_time = time.time()
        face_pred = face_detector.detect_face(image_data)
        inf_time = time.time() - start_time
        data_total_inference_time += inf_time

        ### Calc average IOU, Precision, and Average inferencing time ####
        total_iou = 0
        tp = 0
        pred_dict = dict()
        for gt in face_bbs_gt:
            max_iou_per_gt = 0
            cv2.rectangle(image_data, (gt[0], gt[1]), (gt[2], gt[3]),
                          (255, 0, 0), 2)
            for i, pred in enumerate(face_pred):
                if i not in pred_dict.keys():
                    pred_dict[i] = 0
                cv2.rectangle(image_data, (pred[0], pred[1]),
                              (pred[2], pred[3]), (0, 0, 255), 2)
                iou = get_iou(gt, pred)
                if iou > max_iou_per_gt:
                    max_iou_per_gt = iou
                if iou > pred_dict[i]:
                    pred_dict[i] = iou
            total_iou = total_iou + max_iou_per_gt

        if total_gt_face != 0:
            if len(pred_dict.keys()) > 0:
                for i in pred_dict:
                    if pred_dict[i] >= 0.5:
                        tp += 1
                precision = float(tp) / float(total_gt_face)

            else:
                precision = 0

            image_average_iou = total_iou / total_gt_face
            image_average_precision = precision

            data_total_iou += image_average_iou
            data_total_precision += image_average_precision

    result = dict()
    result['average_iou'] = float(data_total_iou) / float(total_data)
    result['mean_average_precision'] = float(data_total_precision) / float(
        total_data)
    result['average_inferencing_time'] = float(
        data_total_inference_time) / float(total_data)

    return result


def get_args():
    parser = argparse.ArgumentParser(
        description=
        "This script is used to evaluate the face detector on WIDER face dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        required=True,
        help=
        "Please choose between : opencv_haar , dlib_hog , dlib_cnn , mtcnn , mobilenet_ssd "
    )
    parser.add_argument(
        "--iou_threshold",
        "-t",
        type=float,
        default=0.5,
        help=
        "IOU threshold used to determine whether the prediction is matched the ground truth, should be float between 0 and 1"
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    splits = ['train', 'val']
    iou_threshold = args.iou_threshold

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
        data_dict = extract_and_filter_data(splits)

        result = evaluate(face_detector, data_dict, iou_threshold)

        print 'Average IOU = %s' % (str(result['average_iou']))
        print 'mAP = %s' % (str(result['mean_average_precision']))
        print 'Average inference time = %s' % (
            str(result['average_inferencing_time']))


if __name__ == '__main__':
    main()
