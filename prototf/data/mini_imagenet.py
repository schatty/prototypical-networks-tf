import os
import numpy as np
import pickle
from PIL import Image
from functools import partial
import tensorflow as tf


def load_episode(inds, data, n_way, n_support, n_query):
    inds = tf.range(data.shape[0])
    inds = tf.random.shuffle(inds)
    class_way_inds = tf.gather(inds, tf.range(n_way))

    class_img = tf.gather(data, class_way_inds)

    shot_inds = tf.random.shuffle(tf.range(data.shape[1]))
    support_inds = tf.gather(shot_inds, tf.range(n_support))
    query_inds = tf.gather(shot_inds, tf.range(n_support, n_support+n_query))

    support = tf.gather(class_img, support_inds, axis=1)
    query = tf.gather(class_img, query_inds, axis=1)

    return support, query


def load_mini_imagenet(data_dir, config, splits):
    """
    Load miniImagenet dataset.

    Args:
        data_dir (str): path of the directory with 'splits', 'data' subdirs.
        config (dict): general dict with program settings.
        splits (list): list of strings 'train'|'val'|'test'

    Returns (dict): dictionary with keys as splits and values as tf.Dataset

    """
    ret = {}
    for split in splits:
        # n_way (number of classes per episode)
        if split in ['val', 'test']:
            n_way = config['data.test_way']
        else:
            n_way = config['data.train_way']

        # n_support (number of support examples per class)
        if split in ['val', 'test']:
            n_support = config['data.test_n_support']
        else:
            n_support = config['data.train_n_support']

        # n_query (number of query examples per class)
        if split in ['val', 'test']:
            n_query = config['data.test_n_query']
        else:
            n_query = config['data.train_n_query']

        # Load images as numpy
        ds_filename = os.path.join(data_dir, 'data',
                                   f'mini-imagenet-cache-{split}.pkl')
        # load dict with 'class_dict' and 'image_data' keys
        with open(ds_filename, 'rb') as f:
            data_dict = pickle.load(f)
        # Convert original data to format [n_classes, n_img, w, h, c]
        first_key = list(data_dict['class_dict'])[0]
        data = np.zeros((len(data_dict['class_dict']), len(data_dict['class_dict'][first_key]), 84, 84, 3))
        for i, (k, v) in enumerate(data_dict['class_dict'].items()):
                data[i, :, :, :, :] = data_dict['image_data'][v, :]
        data = tf.constant(data / 255., dtype=tf.float16)
        img_ds = tf.data.Dataset.from_tensor_slices(np.arange(n_support, dtype=np.float16))
        img_ds = img_ds.map(
            partial(load_episode, data=data, n_way=n_way, n_support=n_support,
                    n_query=n_query))
        ret[split] = img_ds

    return ret
