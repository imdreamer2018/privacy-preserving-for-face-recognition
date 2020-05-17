import matplotlib.pyplot as plt
import numpy as np

from openface.align import AlignDlib
from openface.openface_model import create_model
from openface.preprocess_face_data import load_image
from openface.preprocess_face_data import load_metadata

# using pre-trained model
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('../models/nn4.small2.v1.h5')

# load customDataset
metadata = load_metadata('../images')

# Initialize the OpenFace face alignment utility
aligment = AlignDlib('../models/landmarks.dat')

# load embedded
embedded = np.load('../face_embedded_data/embedded_nn4_1.npy')


def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))


def show_pair(idx1, idx2):
    plt.figure(figsize=(8, 3))
    plt.suptitle(f'Distance = {distance(embedded[idx1], embedded[idx2]):.2f}',fontsize=15)
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()))
    plt.savefig('../test_images/show_pair_image_distance_1.png')
    plt.show()

#show_pair(12, 0)
show_pair(12, 14)
