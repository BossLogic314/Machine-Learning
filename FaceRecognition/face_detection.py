import cv2
import numpy as np
import os

# Capturing the user's web cam
camera = cv2.VideoCapture(0)

# Creating a classifier object
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# The name of the file where data is stored
file_name = "saved_data.npy"

# Attributes boxes around the faces
color = (0, 255, 255)
thickness = 2

# To store the data
names = []
pixels = []

# Finding out the name of the user to store the data
name = input("Enter the name of the user: ")

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

            rectangle = cv2.rectangle(frame, (x, y), (x + width, y + height), color, thickness)
            cv2.imshow("Screen", rectangle)
            cv2.imshow("Cut screen", face_cut_gray)

    # Waiting for the user to enter something
    key = cv2.waitKey(1)

    # Quit the loop when the user enters 'x'
    if key == ord("x"):
        break

    # Capture the frame if the user enters 'c'
    if key == ord("c"):
        names.append([name])
        pixels.append(face_cut_gray_flat)

data = np.hstack([names, pixels])

# If the file name to store the data already exists, retrieve the stored data restore the new data
if os.path.exists(file_name):

    # Retrieving the stored data
    stored_data = np.load(file_name)

    # Updating the data
    data = np.vstack(data, stored_data)

# Saving the new data
if (names != []) and (pixels != []):
    np.save(file_name, data)

camera.release()
cv2.destroyAllWindows()