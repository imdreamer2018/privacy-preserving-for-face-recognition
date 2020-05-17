from datetime import datetime
import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from pylab import mpl
from sklearn.preprocessing import LabelEncoder
from mpc.nn import Dense, Softmax, CrossEntropy, Sequential, DataLoader
from mpc.tensor import NativeTensor, PrivateEncodedTensor
import os
import config

mpl.rcParams['font.sans-serif'] = ['SimHei']
tf.logging.set_verbosity(tf.logging.ERROR)
np.random.seed(1)
tf.set_random_seed(1)


def accuracy(classifier, x, y, verbose=0, wrapper=NativeTensor):
    predicted_classes = classifier \
        .predict(DataLoader(x, wrapper), verbose=verbose).reveal() \
        .argmax(axis=1)
    correct_classes = NativeTensor(y) \
        .argmax(axis=1)

    matches = predicted_classes.unwrap() == correct_classes.unwrap()
    return sum(matches) / len(matches)


def fineTuneUsingPrivateEncodedTensor():
    classifier.initialize()
    print('start train')
    start = datetime.now()
    classifier.fit(
        DataLoader(X_train, wrapper=PrivateEncodedTensor),
        DataLoader(y_train, wrapper=PrivateEncodedTensor),
        batch_size=1,
        loss=CrossEntropy(),
        epochs=40,
        learning_rate=.03,
        verbose=1
    )
    stop = datetime.now()
    print('Elapsed:', stop - start)

    np.save(weightsPath+'/encrypted_layer0_weights_0.npy', classifier.layers[0].weights.shares0)
    np.save(weightsPath+'/encrypted_layer0_weights_1.npy', classifier.layers[0].weights.shares1)
    np.save(weightsPath+'/encrypted_layer0_bias_0.npy', classifier.layers[0].bias.shares0)
    np.save(weightsPath+'/encrypted_layer0_bias_1.npy', classifier.layers[0].bias.shares1)

    train_accuracy = accuracy(classifier, X_train, y_train)
    test_accuracy = accuracy(classifier, X_test, y_test)

    print("Train accuracy:", train_accuracy)
    print("Test accuracy:", test_accuracy)


if __name__ == '__main__':
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
    y = to_categorical(y, num_labels)

    train_idx = np.arange(metadata.shape[0]) % 5 != 0
    test_idx = np.arange(metadata.shape[0]) % 5 == 0
    print(train_idx,test_idx)
    # 50 train examples of 10 identities (5 examples each)
    X_train = embedded[train_idx]
    print(X_train)
    # 50 test examples of 10 identities (5 examples each)
    X_test = embedded[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]
    # X_train, X_test, y_train, y_test = train_test_split(embedded, y, test_size=0.5)
    classifier = Sequential([
        Dense(num_labels, 128),
        # Reveal(),
        Softmax()
    ])
    weightsPath = config.weights
    if not os.path.exists(weightsPath):
        os.makedirs(weightsPath, exist_ok='True')
    fineTuneUsingPrivateEncodedTensor()
