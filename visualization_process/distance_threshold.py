import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from openface.align import AlignDlib
from openface.openface_model import create_model
from openface.preprocess_face_data import load_image, load_metadata

# using pre-trained model
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('../models/nn4.small2.v1.h5')


# Initialize the OpenFace face alignment utility
aligment = AlignDlib('../models/landmarks.dat')


# load embedded
embedded = np.load('../face_embedded_data/paper/embedded_nn4_lfw_10.npy')

# load customDataset
metadata = np.load('../face_embedded_data/paper/metadata_nn4_lfw_10.npy')

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))


def show_pair(idx1, idx2):
    plt.figure(figsize=(8, 3))
    plt.suptitle(f'Distance = {distance(embedded[idx1], embedded[idx2]):.2f}')
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()))
    plt.show()


distances = []  # squared L2 distance between pairs
identical = []  # 1 if same identity, 0 otherwise

num = embedded.shape[0]

for i in range(num - 1):
    for j in range(1, num):
        distances.append(distance(embedded[i], embedded[j]))
        identical.append(1 if metadata[i].name == metadata[j].name else 0)

distances = np.array(distances)
identical = np.array(identical)

thresholds = np.arange(0.3, 1.0, 0.01)

f1_scores = [f1_score(identical, distances < t) for t in thresholds]
acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]

opt_idx = np.argmax(f1_scores)
# Threshold at maximal F1 score
opt_tau = thresholds[opt_idx]
# Accuracy at maximal F1 score
opt_acc = accuracy_score(identical, distances < opt_tau)

# Plot F1 score and accuracy as function of distance threshold
plt.plot(thresholds, f1_scores, label='F1 score')
plt.plot(thresholds, acc_scores, label='Accuracy')
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title(f'Accuracy and F1 score at threshold {opt_tau:.2f} = {opt_acc:.3f} and {np.max(f1_scores):.3f}')
plt.xlabel('Distance threshold')
plt.legend()
plt.show()
dist_pos = distances[identical == 1]
dist_neg = distances[identical == 0]

plt.subplot(121)
plt.hist(dist_pos)
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title('Distances (pos. pairs)')
plt.legend()

plt.subplot(122)
plt.hist(dist_neg)
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title('Distances (neg. pairs)')
plt.legend()
plt.text(0, 1, 'put some text')
plt.show()
