import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


def calc_euclidian_dists(x, y):
    n = x.shape[0]
    m = y.shape[0]
    x = tf.tile(tf.expand_dims(x, 1), [1, m, 1])
    y = tf.tile(tf.expand_dims(y, 0), [n, 1, 1])
    return tf.reduce_sum(tf.math.pow(x - y, 2), 2)


class Prototypical(Model):
    def __init__(self):
        super(Prototypical, self).__init__()

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
        n_support = support.shape[1]
        n_query = query.shape[1]
        w = support.shape[2]
        h = support.shape[3]
        c = support.shape[4]

        target_inds = tf.reshape(tf.range(n_class), [n_class, 1])
        target_inds = tf.tile(target_inds, [1, n_support])

        cat = tf.concat([
            tf.reshape(support, [n_class * n_support, w, h, c]),
            tf.reshape(query, [n_class * n_query, w, h, c])], axis=0)

        z = self.encoder(cat)
        z_prototypes = tf.reshape(z[:n_class * n_support],
                                  [n_class, n_support, z.shape[-1]])
        z_prototypes = tf.math.reduce_mean(z_prototypes, axis=1)
        z_query = z[n_class * n_support:]

        dists = calc_euclidian_dists(z_query, z_prototypes)

        log_p_y = tf.nn.log_softmax(-dists, axis=-1)
        log_p_y = tf.reshape(log_p_y, [n_class, n_query, -1])

        inds = [[i, i // n_support] for i in list(range(n_class * n_query))]

        loss = -tf.gather_nd(tf.reshape(log_p_y, [n_class * n_query, -1]),
                             inds)
        loss = tf.reshape(loss, [n_class, n_query])
        loss_mean = tf.reduce_mean(loss)

        y_pred = tf.math.argmax(log_p_y, axis=2)
        eq = tf.math.equal(tf.cast(y_pred, tf.int32),
                           tf.cast(target_inds, tf.int32))
        eq = tf.cast(eq, np.int32)
        acc = tf.reduce_sum(eq) / (n_class * n_query)

        return loss_mean, acc

    def save(self, path):
        self.encoder.save(path)