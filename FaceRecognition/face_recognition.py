# Run this file only when some data is stored
import cv2
import sys
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# The name of the file where data is stored
file_name = "saved_data.npy"

# Checking if training data exists
if not os.path.exists(file_name):
    print("Training data does not exist. Please check again.")
    sys.exit()

# Capturing the user's web cam
camera = cv2.VideoCapture(0)

# Creating a classifier object
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Attributes boxes around the faces
color = (0, 255, 255)
thickness = 2
font_scale = 1

# Loading the stored data
data = np.load(file_name)

# Extracting the stored data
names = data[:, 0]
pixels = data[:, 1:]

# Creating a model for face recognition
model = KNeighborsClassifier()

# Training the model with the training data
model.fit(pixels, names)

while True:
    # Reading from the webcam
    status, frame = camera.read()

    if status:
        cv2.imshow("Screen", frame)

        # Detecting faces in the frame
        faces = classifier.detectMultiScale(frame)

        # Drawing a box around each face in the frame
        for face in faces:

            # Storing the coordinates where the box needs to be drawn
            x, y, width, height = face

            # Cutting the face in the image
            face_cut = frame[y:y + height, x:x + width]

            # Storing the gray form of the image
            face_cut_gray = cv2.cvtColor(face_cut, cv2.COLOR_BGR2GRAY)

            # Resizing the face cut region
            face_cut_gray = cv2.resize(face_cut_gray, (200, 200))

            # Flattening the gray image
            face_cut_gray_flat = face_cut_gray.flatten()

            # Predicting the people in the image
            prediction = model.predict([face_cut_gray_flat])
            print(prediction)

            cv2.putText(frame, prediction[0], (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, font_scale, color, thickness)

            rectangle = cv2.rectangle(frame, (x, y), (x + width, y + height), color, thickness)
            cv2.imshow("Screen", rectangle)

    # Waiting for the user to enter something
    key = cv2.waitKey(1)

    # Quit the loop when the user enters 'x'
    if key == ord("x"):
        break

camera.release()
cv2.destroyAllWindows()