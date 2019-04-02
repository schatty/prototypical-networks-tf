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

        inds = [[i, i//n_support] for i in list(range(n_class * n_query))]

        loss = -tf.gather_nd(tf.reshape(log_p_y, [n_class*n_query, -1]), inds)
        loss = tf.reshape(loss, [n_class, n_query])
        loss_mean = tf.reduce_mean(loss)

        y_pred = tf.math.argmax(log_p_y, axis=2)
        eq = tf.math.equal(tf.cast(y_pred, tf.int32), tf.cast(target_inds, tf.int32))
        eq = tf.cast(eq, np.int32)
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
        self.hooks = {name: lambda state: None
                      for name in ['on_start',
                                   'on_start_epoch',
                                   'on_end_epoch',
                                   'on_start_episode',
                                   'on_end_episode',
                                   'on_end']}

    def loss(self, support, query):
        loss, acc = self.model(support, query)
        return loss, acc

    def train(self, train_loader, val_loader, **kwargs):
        self.model = Prototypical()
        optimizer = tf.keras.optimizers.Adam(self.config['train.lr'])

        train_loss = tf.metrics.Mean(name='train_loss')
        val_loss = tf.metrics.Mean(name='val_loss')
        train_acc = tf.metrics.Mean(name='train_accuracy')
        val_acc = tf.metrics.Mean(name='val_accuracy')

        state = {
            'model': self.model,
            'epoch': 1,
            'total_episode': 1,
        }

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
        self.hooks['on_start'](state)
        for epoch in range(n_epochs):
            self.hooks['on_start_epoch'](state)
            for i_episode, (support, query) in enumerate(train_loader):
                self.hooks['on_start_episode'](state)
                train_step(support, query)
                if i_episode+1 == n_episodes:
                    break
                self.hooks['on_end_episode'](state)
                state['total_episode'] += 1
            for support, query in val_loader:
                val_step(support, query)

            template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
            print(template.format(epoch + 1, train_loss.result(), train_acc.result(),
                                  val_loss.result(),
                                  val_acc.result() * 100))
            self.hooks['on_end_episode'](train)
            state['epoch'] += 1

        self.hooks['on_end'](state)
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

    train_engine = TrainEngine(config)

    # Set hooks on training process
    def on_start(state):
        print("Training started.")
    train_engine.hooks['on_start'] = on_start

    def on_end(state):
        print("Training ended.")
    train_engine.hooks['on_end'] = on_end

    def on_start_epoch(state):
        print(f"Epoch {state['epoch']} started.")
    train_engine.hooks['on_start_epoch'] = on_start_epoch

    def on_end_epoch(state):
        print(f"Epoch {state['epoch']} ended.")
    train_engine.hooks['on_end_epoch'] = on_end_epoch

    def on_start_episode(state):
        print(f"Episode {state['total_episode']}")
    train_engine.hooks['on_start_episode'] = on_start_episode

    def on_end_episode(state):
        return None
    train_engine.hooks['on_end_episode'] = on_end_episode

    with tf.device(device_name):
        train_engine.train(
            model=None,
            train_loader=train_loader,
            val_loader=val_loader,
            optim_method=config['train.optim_method'],
            optim_config={'lr': config['train.lr'],
                          'weight_decay': config['train.weight_decay']},
            max_epoch=config['train.epochs']
        )
