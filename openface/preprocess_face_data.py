import numpy as np
import os.path
import cv2
from openface.align import AlignDlib
from openface.openface_model import create_model


class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)


def load_metadata(path, num):
    metadata = []
    for i in sorted(os.listdir(path))[:num]:
        for f in sorted(os.listdir((os.path.join(path, i)))):
            # Check file extension. Allow only jpg/jpeg/png file
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == 'JPG' or ext == 'jpeg' or ext == 'png':
                metadata.append(IdentityMetadata(path, i, f))

    return np.array(metadata)


def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[..., ::-1]


def embeddingVectors(path):
    # using pre-trained model
    nn4_small2_pretrained = create_model()
    nn4_small2_pretrained.load_weights('models/nn4.small2.v1.h5')

    # Initialize the OpenFace face alignment utility
    aligment = AlignDlib('models/landmarks.dat')

    # Align image on face
    def align_image(img):
        return aligment.align(96, img, aligment.getLargestFaceBoundingBox(img),
                              landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

    img = load_image(path)
    img = align_image(img)
    try:
        # scale RGB values to interval [0,1]
        img = (img / 255.).astype(np.float32)
    except TypeError:
        print("The image is not Clear to extract the Embeddings")
    else:
        # obtain embedding vector for image
        return nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]



