import numpy as np
from prototf.data import load

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


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
        print("Support shape: ", support.shape)

        n_class = support.shape[0]
        n_support = support.shape[1]
        n_query = query.shape[1]
        w = support.shape[2]
        h = support.shape[3]
        c = support.shape[4]

        print("n_class: ", n_class)

        target_inds = tf.reshape(tf.range(n_class), [n_class, 1])
        print("Target inds original: ", target_inds.shape)
        target_inds = tf.tile(target_inds, [1, n_support])
        print("Target inds: ", target_inds.shape)

        print(support.shape, query.shape)

        cat = tf.concat([
            tf.reshape(support, [n_class * n_support, w, h, c]),
            tf.reshape(query, [n_class * n_query, w, h, c])], axis=0)
        print("cat: ", cat.shape)

        z = self.encoder(cat)
        print("z: ", z.shape)
        z_prototypes = tf.reshape(z[:n_class * n_support],
                                  [n_class, n_support, z.shape[-1]])
        z_prototypes = tf.math.reduce_mean(z_prototypes, axis=1)
        print("z_prototypes: ", z_prototypes.shape)

        z_query = z[n_class * n_support:]
        print("z_query: ", z_query.shape)

        dists = calc_euclidian_dists(z_query, z_prototypes)

        log_p_y = tf.nn.log_softmax(-dists, axis=-1)
        log_p_y = tf.reshape(log_p_y, [n_class, n_query, -1])
        print("Log Prob: ", log_p_y.shape)

        inds = [[i, i//n_support] for i in list(range(n_class * n_query))]
        print("INDS: ", len(inds), len(inds[0]))

        loss = -tf.gather_nd(tf.reshape(log_p_y, [n_class*n_query, -1]), inds)
        loss = tf.reshape(loss, [n_class, n_query])
        print("loss: ", loss.shape)
        loss_mean = tf.reduce_mean(loss)

        y_pred = tf.math.argmax(log_p_y, axis=2)
        eq = tf.math.equal(tf.cast(y_pred, tf.int32), tf.cast(target_inds, tf.int32))
        eq = tf.cast(eq, np.int32)
        print("eq: ", eq.shape)
        acc = tf.reduce_sum(eq) / (n_class * n_query)

        return loss_mean, acc


def calc_euclidian_dists(x, y):
    n = x.shape[0]
    m = y.shape[0]
    x = tf.tile(tf.expand_dims(x, 1), [1, m, 1])
    y = tf.tile(tf.expand_dims(y, 0), [n, 1, 1])
    return tf.reduce_sum(tf.math.pow(x - y, 2), 2)


class TrainEngine(object):
    def __init__(self, config):
        self.config = config

    def loss(self, support, query):
        loss, acc = self.model(support, query)
        return loss, acc

    def train(self, train_loader, val_loader, **kwargs):
        self.model = Prototypical()

        train_loss = tf.metrics.Mean(name='train_loss')
        val_loss = tf.metrics.Mean(name='val_loss')
        train_acc = tf.metrics.Mean(name='train_accuracy')
        val_acc = tf.metrics.Mean(name='val_accuracy')
        optimizer = tf.keras.optimizers.Adam(self.config['train.lr'])

        @tf.function
        def train_step(support, query):
            # Forward & update gradients
            with tf.GradientTape() as tape:
                loss, acc = self.loss(support, query)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))

            # Log loss and accuracy for step
            train_loss(loss)
            train_acc(acc)

        @tf.function
        def val_step(support, query):
            loss, acc = self.loss(support, query)
            val_loss(loss)
            val_acc(acc)

        n_episodes = self.config['data.train_episodes']
        n_epochs = self.config['train.epochs']
        for epoch in range(n_epochs):
            for i_episode, (support, query) in enumerate(train_loader):
                train_step(support, query)
                if i_episode+1 == n_episodes:
                    break
            for support, query in val_loader:
                val_step(support, query)

            template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
            print(template.format(epoch + 1, train_loss.result(), train_acc.result(),
                                  val_loss.result(),
                                  val_acc.result() * 100))

        print("Success!")


def train(config):
    print("Config: ", config)
    #data_dir = '/home/igor/dl/prototypical-networks/data/omniglot'
    data_dir = '/Users/sirius/code/prototypical-networks-tf/data/omniglot'
    ret = load(data_dir, config, ['train', 'val'])
    train_loader = ret['train']
    val_loader = ret['val']

    # Determine device
    if config['data.cuda']:
        device_name = 'GPU:0'
    else:
        device_name = 'CPU:0'

    experiment = TrainEngine(config)
    with tf.device(device_name):
        experiment.train(
            model=None,
            train_loader=train_loader,
            val_loader=val_loader,
            optim_method=config['train.optim_method'],
            optim_config={'lr': config['train.lr'],
                          'weight_decay': config['train.weight_decay']},
            max_epoch=config['train.epochs']
        )
