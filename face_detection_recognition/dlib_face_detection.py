import face_recognition
import cv2

# Load the jpg file into a numpy array
image = cv2.imread("test.jpg")

# Find all the faces in the image
face_locations = face_recognition.face_locations(image)

# Get number of faces
number_of_faces = len(face_locations)
print("Number of face(s) in this image {}.".format(number_of_faces))

for face_location in face_locations:
    # Get coord
    x, y, z, w = face_location

    # Draw Face rectangle
    cv2.rectangle(image,(w,x),(y,z),(0,0,255),2)

# Show image
cv2.imshow("img",image)
cv2.imwrite("detected.jpg",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
