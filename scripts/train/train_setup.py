import numpy as np
from prototf.data import load

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


class Prototypical(Model):
    def __init__(self):
        super(Prototypical, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)), Flatten()]
        )

    # def encoder(self, x):
    #     #x = self.conv1(x)
    #     #x = self.flatten(x)
    #     #x = self.d1(x)
    #     #return self.d2(x)
    #     return tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
    #         tf.keras.layers.BatchNormalization(),
    #         tf.keras.layers.ReLU(),
    #         tf.keras.layers.MaxPool2D((2, 2)), Flatten()]
    #     )(x)

    def call(self, support, query):
        n_class = support.shape[0]
        n_support = support.shape[1]
        n_query = query.shape[1]
        w = support.shape[2]
        h = support.shape[3]
        c = support.shape[4]

        target_inds = tf.reshape(tf.range(n_class), [n_class, 1])
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

        soft = tf.nn.softmax(dists, axis=-1)
        soft = tf.reshape(soft, [n_class, n_query, -1])
        print("Softmax: ", soft.shape)

        #inds = np.arange(n_class)
        #loss = -tf.gather(soft, inds, axis=2)
        #print("loss: ", loss.shape)
        #loss_mean = tf.reduce_mean(loss)

        inds = [[i, i//n_support] for i in list(range(n_class * n_query))]
        print("INDS: ", len(inds), len(inds[0]))

        loss = -tf.gather_nd(tf.reshape(soft, [n_class*n_query, -1]), inds)
        loss = tf.reshape(loss, [n_class, n_query])
        print("loss: ", loss.shape)
        loss_mean = tf.reduce_mean(loss)

        y_pred = tf.math.argmax(soft, axis=-1)
        eq = tf.math.equal(tf.cast(y_pred, tf.int32), target_inds)
        eq = tf.cast(eq, np.int32)
        print("eq: ", eq.shape)
        acc = tf.reduce_mean(eq)

        return loss_mean, acc


def calc_euclidian_dists(x, y):
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[0]

    x = tf.tile(tf.expand_dims(x, 1), [1, m, 1])
    y = tf.tile(tf.expand_dims(y, 0), [n, 1, 1])

    print("x: ", x.shape)
    print("y: ", y.shape)

    res = tf.reduce_sum(tf.math.pow(x - y, 2), -1)
    print("Euclid result: ", res.shape)

    return res


class Experiment(object):
    def __init__(self, config):
        self.config = config

    def train(self, loader, **kwargs):

        model = Prototypical()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        acc_monitor = tf.keras.metrics.Mean(name='train_accuracy')
        #loss_object = tf.losses.LogLoss()
        optimizer = tf.keras.optimizers.Adam()

        @tf.function
        def train_step(support, query):
            with tf.GradientTape() as tape:
                loss, acc = model(support, query)
                #loss = loss_object(np.ones(y_pred.shape[0]), y_pred)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))

            train_loss(loss)
            acc_monitor(acc)

            #train_accuracy(label, predictions)

        # @tf.function
        # def test_step(support, query):
        #     predictions = model(support, query)
        #     t_loss = loss_object(label, predictions)
        #
        #     test_loss(t_loss)
        #     test_accuracy(label, predictions)

        n_epochs = 100
        sss = 0
        for epoch in range(n_epochs):
            for support, query in loader:
                train_step(support, query)
                print("Step ", sss)
                sss += 1
                #break
            # for test_image, test_label in test_loader:
            #     test_step(test_image, test_label)

            #template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            template = 'Epoch {}, Loss: {}, Acc: {}'
            print(template.format(epoch + 1, train_loss.result(), acc_monitor.result()))
                                  #test_loss.result(),
                                  #test_accuracy.result() * 100))

        print("Success!")

        '''

        # Losses and metrics monitors
        
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(
            name='test_accuracy')

        '''


def train(config):
    print("Config: ", config)
    data_dir = './data/omniglot'
    ret = load(data_dir, config, ['train'])
    img_ds = ret['train']
    img_ds = img_ds.batch(config['data.train_way'])

    experiment = Experiment(config)
    experiment.train(
        model=None,
        loader=img_ds,
        optim_method=config['train.optim_method'],
        optim_config={'lr': config['train.lr'],
                      'weight_decay': config['train.weight_decay']},
        max_epoch=config['train.epochs']
    )