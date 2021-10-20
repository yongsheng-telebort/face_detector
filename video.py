# Phase 1: Import OpenCV
import cv2 as cv

# Phase 2: Load trained data
trained_face_data = cv.CascadeClassifier(
    "haarcascade_frontalface_default.xml")

# Phase 3: Image Processing for Detection
# capture video
video = cv.VideoCapture(0)

while True:
    # capture frame-by-frame
    ret, frame = video.read()
    # Change image to gray
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("Gray", gray_img)

    # Phase 4: Face Detection
    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(gray_img)

    # To draw rectangle for each face
    for face in face_coordinates:
        (x, y, w, h) = face
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Quit when no more frame
    if not ret:
        print("Video Ended")
        break
    # show video frame-by-frame
    cv.imshow('frame', frame)
    # stop when Q is pressed
    if cv.waitKey(25) == ord("q"):
        break

# Release everything if job is finished
video.release()
cv.destroyAllWindows()
