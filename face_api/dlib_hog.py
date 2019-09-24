import face_recognition
import pandas as pd
import os


class DlibHOGFaceDetector():

    def __init__(self):

        # Relative dataset path to this module
        self.dataset_path = os.path.join(os.path.dirname(__file__), 'data/data.pkl')

        # Relative dataset images path to this module
        self.face_images_path = os.path.join(os.path.dirname(__file__), 'data/dataset')

    def face_detector(self, img_path):
        """
        Face detection using Dlib
        :param img: input image
        :return: faces bboxes
        """

        # Load image
        img = face_recognition.load_image_file(img_path)

        # Find all the faces in the image
        face_locations = face_recognition.face_locations(img)

        return face_locations

    def dlib_recognition(self, img_path):
        """
        Face recognition using Dlib
        :param img: input image
        :return: Names of the detected persons
        """
        output_name = "no body !"

        # Load dataset
        dataset = pd.read_pickle(self.dataset_path)

        encodings = dataset["encodings"].values
        names = dataset["names"].values
        # Load image
        img = face_recognition.load_image_file(img_path)

        # Encode facial features
        encoding = face_recognition.face_encodings(img)[0]
        i = 0

        for encode in encodings:

            result = face_recognition.compare_faces([encode], encoding)

            if result[0] == True:
                output_name = names[i]
                break

            i += 1

        return output_name

    def create_dataset(self):

        # Faces encoding features
        features = []

        # Persons names extracted from images names
        names = []

        # Filter all files in the directory
        for filename in os.listdir(self.face_images_path):

            # Make sure that our file is text
            if (filename.endswith('.jpeg')) or (filename.endswith('.jpg')) or (filename.endswith('.png')):

                # Load image
                image = face_recognition.load_image_file(self.face_images_path + "/" + filename)

                # Encode face features
                feature = face_recognition.face_encodings(image)[0]

                # Appen feature to total features list
                features.append(feature)

                # Add person name to dataset
                names.append(os.path.splitext(filename)[0])

                print("<<<<<<<< Encoded >>>>>>>>>>> <<<<<<<< " + os.path.splitext(filename)[0] + " >>>>>>>>>>>")

        # Save dataset
        data = pd.DataFrame({"encodings": features, "names": names})
        pd.to_pickle(data, self.dataset_path)

    def add_face(self, image_path):

        # Load dataset
        dataset = pd.read_pickle(self.dataset_path)

        # Get old data
        names = list(dataset['names'].values)
        features = list(dataset['encodings'].values)

        # Get file name
        filename = os.path.basename(image_path)

        # Person name
        name = os.path.splitext(filename)[0]

        # Check if the person already exists
        exist = False
        for n in names:
            if n == name:
                exist = True
                break

        if(exist == False):
            # Add name to other names
            names.append(name)

            # Load image
            image = face_recognition.load_image_file(image_path)

            # Encode face features
            feature = face_recognition.face_encodings(image)[0]

            # Append feature to total features list
            features.append(feature)

            # Save dataset
            data = pd.DataFrame({"encodings": features, "names": names})
            pd.to_pickle(data, self.dataset_path)

            print(os.path.splitext(filename)[0] + " Added successefuly !")
        else:
            print(name + " already exist !")