from openface.triplet_generator_data import triplet_generator
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Layer
from openface.openface_model import create_model

nn4_small2 = create_model()

# Input for anchor, positive and negtive images
in_a = Input(shape=(96, 96, 3))
in_p = Input(shape=(96, 96, 3))
in_n = Input(shape=(96, 96, 3))

# Output for anchor, positive and negative embedding vectors
# The nn4_small model instance is shared (Siamese network)
emb_a = nn4_small2(in_a)
emb_p = nn4_small2(in_p)
emb_n = nn4_small2(in_n)


class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        a, p, n = inputs
        p_dist = K.sum(K.square(a - p), axis=1)
        n_dist = K.sum(K.square(a - n), axis=1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs, **kwargs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss


# Layer that computes the triplet loss from anchor, positive and negative embedding vectors
triplet_loss_layer = TripletLossLayer(alpha=0.2, name='triple_loss_layer')([emb_a, emb_p, emb_n])

# Model that can be trained with anchor, positive negative images
nn4_small2_train = Model([in_a, in_p, in_n], triplet_loss_layer)

# triplet_generator() creates a generator that continuously returns
# ([a_batch, p_batch, n_batch], None) tuples where a_batch, p_batch
# and n_batch are batches of anchor, positive and negative RGB images
# each having a shape of (batch_size, 96, 96, 3).
generator = triplet_generator()

nn4_small2_train.compile(loss=None, optimizer='adam')
nn4_small2_train.fit_generator(generator,epochs=10,steps_per_epoch=100)
# Please note that the current implementation of the generator only generates
# random image data. The main goal of this code snippet is to demonstrate
# the general setup for model training. In the following, we will anyway
# use a pre-trained model so we don't need a generator here that operates
# on real training data. I'll maybe provide a fully functional generator
# later.
