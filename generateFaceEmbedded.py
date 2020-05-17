from openface.openface_model import create_model
from openface.preprocess_face_data import load_metadata
from openface.align import AlignDlib
import numpy as np
import cv2
import config
import os
from datetime import datetime

# using pre-trained model
print('load_model')
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('models/nn4.small2.v1.h5')

# nn4_small2_pretrained.summary()
start = datetime.now()
# load customDataset
metadata = load_metadata(config.faceImagesPath, num=50)
print('load_image')

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[..., ::-1]

# Initialize the OpenFace face alignment utility
aligment = AlignDlib('models/landmarks.dat')

# Align image on face
def align_image(img):
    return aligment.align(96, img, aligment.getLargestFaceBoundingBox(img),
                          landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

# Embedding vectors
good_image_index = []
unfit_image_index = []
embedded = np.zeros((metadata.shape[0], 128))
print('preprocess image')
for i, m in enumerate(metadata):
    img = load_image(m.image_path())
    img = align_image(img)
    try:
        # scale RGB values to interval [0,1]
        img = (img / 255.).astype(np.float32)
    except TypeError:
        unfit_image_index.append(i)
        print("The image is not Clear to extract the Embeddings")
    else:
        # obtain embedding vector for image
        embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
        good_image_index.append(i)
stop = datetime.now()
print(stop - start)
metadata = metadata[good_image_index]
embedded = embedded[good_image_index]
print('face embedded create complete')
print('save metadata and embedded')
if not os.path.exists(config.faceData):
    os.makedirs(config.faceData, exist_ok='True')
# save metadata
np.save(config.faceData+'/metadata.npy', metadata)
# save embedded
np.save(config.faceData+'/embedded.npy', embedded)
