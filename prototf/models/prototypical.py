import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.models import load_model


def calc_euclidian_dists(x, y):
    n = x.shape[0]
    m = y.shape[0]
    x = tf.tile(tf.expand_dims(x, 1), [1, m, 1])
    y = tf.tile(tf.expand_dims(y, 0), [n, 1, 1])
    return tf.reduce_sum(tf.math.pow(x - y, 2), 2)


class Prototypical(Model):
    def __init__(self, n_support, n_query, w, h, c):
        super(Prototypical, self).__init__()
        
        self.n_class = None
        self.n_support = n_support
        self.n_query = n_query
        self.w, self.h, self.c = w, h, c

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)), Flatten()]
        )

    def call(self, support, query):
        n_class = support.shape[0]
        target_inds = tf.reshape(tf.range(n_class), [n_class, 1])
        target_inds = tf.tile(target_inds, [1, self.n_support])

        cat = tf.concat([
            tf.reshape(support, [n_class * self.n_support,
                                 self.w, self.h, self.c]),
            tf.reshape(query, [n_class * self.n_query,
                               self.w, self.h, self.c])], axis=0)

        z = self.encoder(cat)
        z_prototypes = tf.reshape(z[:n_class * self.n_support],
                                  [n_class, self.n_support, z.shape[-1]])
        z_prototypes = tf.math.reduce_mean(z_prototypes, axis=1)
        z_query = z[n_class * self.n_support:]

        dists = calc_euclidian_dists(z_query, z_prototypes)

        log_p_y = tf.nn.log_softmax(-dists, axis=-1)
        log_p_y = tf.reshape(log_p_y, [n_class, self.n_query, -1])

        inds = [[i, i // self.n_support] for i in list(range(n_class * self.n_query))]

        loss = -tf.gather_nd(tf.reshape(log_p_y, [n_class * self.n_query, -1]),
                             inds)
        loss = tf.reshape(loss, [n_class, self.n_query])
        loss_mean = tf.reduce_mean(loss)

        y_pred = tf.math.argmax(log_p_y, axis=2)
        eq = tf.math.equal(tf.cast(y_pred, tf.int32),
                           tf.cast(target_inds, tf.int32))
        eq = tf.cast(eq, np.int32)
        acc = tf.reduce_sum(eq) / (n_class * self.n_query)

        return loss_mean, acc

    def save(self, model_path):
        self.encoder.save(model_path)

    def load(self, model_path):
        self.encoder(tf.zeros([self.n_support, self.w, self.h, self.c]))
        self.encoder.load_weights(model_path)