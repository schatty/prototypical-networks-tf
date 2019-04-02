"""
Logic for evaluation procedure of saved model.
"""

import tensorflow as tf

from prototf.models import Prototypical
from prototf.data import load


def eval(config):
    model = Prototypical(5, 5, 28, 28, 1)
    model.load("model.h5")
    print("Model loaded.")

    # Determine device
    if config['data.cuda']:
        device_name = 'GPU:0'
    else:
        device_name = 'CPU:0'

    # Load data from disk
    data_dir = 'data/omniglot'
    ret = load(data_dir, config, ['test'])
    test_loader = ret['test']

    # Metrics to gather
    test_loss = tf.metrics.Mean(name='test_loss')
    test_acc = tf.metrics.Mean(name='test_accuracy')

    def calc_loss(support, query):
        loss, acc = model(support, query)
        return loss, acc

    with tf.device(device_name):
        for i_episode, (support, query) in enumerate(test_loader):
            print("Episode: ", i_episode + 1)
            loss, acc = calc_loss(support, query)
            test_loss(loss)
            test_acc(acc)

    print("Loss: ", test_loss.result().numpy())
    print("Accuracy: ", test_acc.result().numpy())


if __name__ == "__main__":
    pass