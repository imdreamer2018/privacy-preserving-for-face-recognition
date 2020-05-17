import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from mpc.nn import Dense, Reveal, Softmax, Sequential
from mpc.tensor import PrivateEncodedTensor
from openface.align import AlignDlib
from openface.openface_model import create_model
import os
import config


import warnings

warnings.filterwarnings("ignore")

# load sample data
def predict(classifier, wrapper, x):
    likelihoods = classifier.predict(wrapper(x), batch_size=128)
    prob = np.max(likelihoods.unwrap())
    print(likelihoods.unwrap())
    if (prob < 0.5):
        return 'Unknown', prob
    y_predicted = np.argmax(likelihoods.unwrap())

    return y_predicted, prob

# using pre-trained model
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('models/nn4.small2.v1.h5')

# Initialize the OpenFace face alignment utility
aligment = AlignDlib('models/landmarks.dat')

# Align image on face
def align_image(img):
    return aligment.align(96, img, aligment.getLargestFaceBoundingBox(img),
                          landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

def img_to_embedding(image):
    img = image[..., ::-1]
    img = align_image(img)
    try:
        # scale RGB values to interval [0,1]
        img = (img / 255.).astype(np.float32)
    except TypeError:
        print("The image is not Clear to extract the Embeddings")
    else:
        # obtain embedding vector for image
        return nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]


face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# load customDataset
metadata = np.load(config.faceData+'/metadata.npy')
# load embedded
embedded = np.load(config.faceData+'/embedded.npy')

targets = np.array([m.name for m in metadata])
encoder = LabelEncoder()
encoder.fit(targets)

# Numerical encoding of identities
y = encoder.transform(targets)
num_labels = len(np.unique(y))

classifier = Sequential([
    Dense(num_labels, 128),
    Reveal(),
    Softmax()
])

classifier.layers[0].weights = PrivateEncodedTensor.from_shares(
    np.load(config.weights+'/encrypted_layer0_weights_0.npy'),
    np.load(config.weights+'/encrypted_layer0_weights_1.npy'))
classifier.layers[0].bias = PrivateEncodedTensor.from_shares(np.load(config.weights+'/encrypted_layer0_bias_0.npy'),
                                                             np.load(config.weights+'/encrypted_layer0_bias_1.npy',))

video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi = frame[y:y + h, x:x + w]
        try:
            example_image_embedded = img_to_embedding(roi).reshape(1, -1)
        except AttributeError:
            continue
        example_identity, prob = predict(classifier, PrivateEncodedTensor, example_image_embedded)
        if (example_identity == 'Unknown'):
            example_identity_ = 'Unknown person'
        else:
            example_identity_ = encoder.inverse_transform(example_identity)
        cv2.putText(frame, "Face : " + str(example_identity_), (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv2.putText(frame, "prob : " + str(prob), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    cv2.imshow('Face Recognition System', frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

video_capture.release()
cv2.destroyAllWindows()
