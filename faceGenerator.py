import os
import cv2
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
import config

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=200)

video_capture = cv2.VideoCapture(0)

name = input("Enter name of person:")

path = config.faceImagesPath
directory = os.path.join(path, name)
number_of_images = 0
if not os.path.exists(directory):
    os.makedirs(directory, exist_ok='True')
else:
    for fn in os.listdir(directory):
        number_of_images += 1
print(number_of_images)
MAX_NUMBER_OF_IMAGES = number_of_images + 30
count = 0

while number_of_images < MAX_NUMBER_OF_IMAGES:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    faces = detector(frame)
    if len(faces) == 1:
        face = faces[0]
        (x, y, w, h) = face_utils.rect_to_bb(face)
        face_img = frame[y - 50:y + h + 100, x - 50:x + w + 100]
        if count == 5:
            cv2.imwrite(os.path.join(directory, str(name + str(number_of_images) + '.jpg')), face_img)
            number_of_images += 1
            count = 0
        print(count)
        count += 1
    cv2.imshow('Video', frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

video_capture.release()
cv2.destroyAllWindows()