import cv2 as cv


# Load Haar Cascade Classifier
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

# Read the image
img = cv.imread('test.jpg')

# Convert image to graysclae
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detec faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv.imwrite("detection.jpg",img)
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()
