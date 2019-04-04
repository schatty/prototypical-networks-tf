"""
Logic for evaluation procedure of saved model.
"""

import tensorflow as tf
tf.config.gpu.set_per_process_memory_growth(True)

from prototf.models import Prototypical
from prototf.data import load


def eval(config):
    n_support = config['data.test_n_support']
    n_query = config['data.test_n_query']
    w, h, c, = list(map(int, config['model.x_dim'].split(',')))
    model = Prototypical(n_support, n_query, w, h, c)
    model_path = f"{config['model.model_path']}"
    model.load(model_path)
    print("Model loaded.")

    # Determine device
    if config['data.cuda']:
        cuda_num = config['data.gpu']
        device_name = f'GPU:{cuda_num}'
    else:
        device_name = 'CPU:0'

    # Load data from disk
    data_dir = f"data/{config['data.dataset']}"
    ret = load(data_dir, config, ['test'])
    test_loader = ret['test']

    # Metrics to gather
    test_loss = tf.metrics.Mean(name='test_loss')
    test_acc = tf.metrics.Mean(name='test_accuracy')

    def calc_loss(support, query):
        loss, acc = model(support, query)
        return loss, acc

    with tf.device(device_name):
        for support, query in test_loader:
            loss, acc = calc_loss(support, query)
            test_loss(loss)
            test_acc(acc)

    print("Loss: ", test_loss.result().numpy())
    print("Accuracy: ", test_acc.result().numpy())


if __name__ == "__main__":
    pass
