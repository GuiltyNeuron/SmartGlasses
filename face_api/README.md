# :man: Face API
Face detection using : 
 - OpenCV Haar Cascade Classifier
 - DLIB face detection using SVM and HOG descriptor
 - DLIB face detection using CNN
 - SSD face detection
 - MTCNN face detector

Face recognition using :
 - DLIB

face_engine classes :
 - OpenCVHaarFaceDetector
     - face_detector() : detect faces with Haar Cascade Classifier
 - DlibHOGFaceDetector
     - face_detector() : detect faces with Dlib based on Hog and SVM classifier
     - dlib_recognition() : Recognise face using Dlib
     - create_dataset() : Create a dataset of faces encodings and their names
     - add_face() : Add new face to Dataset
 - DlibCNNFaceDetector
    - face_detector() : detect faces with Dlib based on CNN
 - TensorflowMTCNNFaceDetector
    - face_detector() : detect faces with CNN : MTCNN
 - TensoflowMobilNetSSDFaceDector
    - face_detector() : detect faces with CNN : ssd mobilenet
# :books: Documentation
- Dlib github Readme file [link](https://github.com/ageitgey/face_recognition)
- Face detection and recognition Raspberry Pi [Link](https://www.pyimagesearch.com/2018/06/25/raspberry-pi-face-recognition/)
- Face detection benchmark [Link](https://github.com/nodefluxio/face-detector-benchmark)
## Licence
GuideMeGlasses
:eyeglasses: