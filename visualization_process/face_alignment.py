import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from openface.align import AlignDlib
from openface.preprocess_face_data import load_image, load_metadata

metadata = load_metadata('../test_image', 1)

#Initialize the OpenFace face alignment utility
aligment = AlignDlib('../models/landmarks.dat')

#Load an image of Jacques Chirac
jc_orig = load_image(metadata[2].image_path())

#Detect face and return bounding box
bb = aligment.getLargestFaceBoundingBox(jc_orig)

#Transform image using specified face landmark indices and crop image to 96 * 96
jc_aligned = aligment.align(96, jc_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

#Show original image
plt.subplot(131)
plt.imshow(jc_orig)

#show original image with bounding box
plt.subplot(132)
plt.imshow(jc_orig)
plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(), fill=False, color='red'))

#show aligned image
plt.subplot(133)
plt.imshow(jc_aligned)

plt.imsave('../test_images/111.png',jc_aligned)
# plt.savefig('../test_images/YangQian_011.png')
plt.show()