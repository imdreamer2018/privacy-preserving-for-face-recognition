import numpy as np
from datetime import datetime

from tensorflow.python.keras.utils import to_categorical
import matplotlib.pyplot as plt
from openface.preprocess_face_data import load_metadata, load_image, embeddingVectors

from mpc.tensor import NativeTensor, PrivateEncodedTensor, PublicEncodedTensor
from mpc.nn import Dense, Reveal, Diff, Softmax, CrossEntropy, Sequential, DataLoader
from sklearn.preprocessing import LabelEncoder


# load sample data
def predict(classifier, wrapper, x):
    likelihoods = classifier.predict(wrapper(x), batch_size=128)
    print(likelihoods)
    prob = np.max(likelihoods.unwrap())
    print(likelihoods.unwrap())
    if (prob < 0.5):
        return 'Unknown', prob
    y_predicted = np.argmax(likelihoods.unwrap())

    return y_predicted, prob


def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))


if __name__ == '__main__':

    # load customDataset
    metadata = np.load('face_embedded_data/paper/metadata_nn4_lfw_10.npy')

    # load embedded
    embedded = np.load('face_embedded_data/paper/embedded_nn4_lfw_10.npy')

    targets = np.array([m.name for m in metadata])
    encoder = LabelEncoder()
    encoder.fit(targets)

    # Numerical encoding of identities
    y = encoder.transform(targets)
    num_labels = len(np.unique(y))

    # define unknow threshold
    unknown_threshold = 0.65

    # load image
    example_image_path = 'test_image/test.jpg'
    example_image = load_image(example_image_path)
    example_image_embedded = embeddingVectors(example_image_path).reshape(1, -1)

    classifier = Sequential([
        Dense(num_labels, 128),
        Reveal(),
        Softmax()
    ])

    classifier.layers[0].weights = PrivateEncodedTensor.from_shares(
        np.load('weights/paper/encrypted_10_layer0_weights_0.npy'),
        np.load('weights/paper/encrypted_10_layer0_weights_1.npy'))
    classifier.layers[0].bias = PrivateEncodedTensor.from_shares(
        np.load('weights/paper/encrypted_10_layer0_bias_0.npy'),
        np.load('weights/paper/encrypted_10_layer0_bias_1.npy'))

    # image with all images in DB of distance
    # distanceInputFromDB = []
    # for j in embedded:
    #     distanceInputFromDB.append(distance(example_image_embedded, j))
    #
    # print(np.sort(distanceInputFromDB))
    # print(np.mean(np.sort(distanceInputFromDB)[0:9]))
    # if np.mean(np.sort(distanceInputFromDB)[0:9]) < unknown_threshold:
    #     example_identity,prob = predict(classifier, PrivateEncodedTensor, example_image_embedded)
    #
    #     if (example_identity == 'Unknown'):
    #         example_identity_ = 'Unknown person'
    #     else:
    #         example_identity_ = encoder.inverse_transform(example_identity)
    #     plt.imshow(example_image)
    #     plt.title(f'Keras Encrypted Model Recognized as {example_identity_}, and prob as {prob}')
    # else:
    #     plt.imshow(example_image)
    #     plt.title(f'Keras Encrypted Model Recognized as Unknown person, and distance as {np.mean(np.sort(distanceInputFromDB)[0:9])}')
    # plt.show()
    start = datetime.now()
    example_identity, prob = predict(classifier, PrivateEncodedTensor, example_image_embedded)
    stop = datetime.now()
    print(stop - start)
    if (example_identity == 'Unknown'):
        example_identity_ = 'Unknown person'
    else:
        example_identity_ = encoder.inverse_transform(example_identity)
    plt.imshow(example_image)
    plt.title(f'Keras Encrypted Model Recognized as {example_identity_}, and prob as {prob}')
    plt.show()