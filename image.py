# Phase 1: Import OpenCV
import cv2 as cv

# Phase 2: Load trained data
trained_face_data = cv.CascadeClassifier(
    "haarcascade_frontalface_default.xml")

# Phase 3: Image Processing for Detection
# Read image
img = cv.imread("outing.jpg")

# resize image
scale = 0.2
width = int(img.shape[1]*scale)
height = int(img.shape[0]*scale)
dim = (width, height)
img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

# Change image to gray
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray_img)

# Phase 4: Face Detection
# Detect faces
face_coordinates = trained_face_data.detectMultiScale(gray_img)

# To draw rectangle for each face
for face in face_coordinates:
    (x, y, w, h) = face
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Phase 5: Show image
cv.imshow("Detected faces", img)
cv.waitKey(0)
