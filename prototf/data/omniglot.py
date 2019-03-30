mport os
import sys
import glob
import random
import pathlib
from functools import partial

import numpy as np
from PIL import Image

import tensorflow as tf

OMNIGLOT_DATA_DIR = os.path.join(__file__, 'data/omniglot')

def class_names_to_paths(class_names):
    d = []
    rots = []
    for class_name in class_names:
        alphabet, character, rot = class_name.split('/')
        image_dir = os.path.join(OMNIGLOT_DATA_DIR, 'data', alphabet, character)
        d.append(image_dir)
        rots.append(rot)
    return d, rots

def get_class_images_paths(dir_paths, rotates):
    classes, img_paths, rotates_list = [], [], []
    for dir_path, rotate in zip(dir_paths, rotates):
        class_images = sorted(glob.glob(os.path.join(dir_path, '*.png')))

        classes.append(dir_path)
        img_paths.append(class_images)
        rotates_list.append(int(rotate[3:]))
    return classes, img_paths, rotates_list


def preprocess_image(img, rot):        
    image = tf.image.decode_png(img, channels=1)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (28, 28))
    image /= 255.0
    return image


def load_and_preprocess_image(img_path, rot):
    image = tf.io.read_file(img_path)
    return preprocess_image(image, rot)

def load_class_images(class_name, img_paths, rot):
    n_examples = img_paths.shape[0]
    example_inds = tf.range(n_examples)
    example_inds = tf.random.shuffle(example_inds)

    support_inds = example_inds[:n_support]
    support_paths = tf.gather(img_paths, support_inds)
    query_inds = example_inds[n_support:]
    query_paths = tf.gather(img_paths, query_inds)

    # Build support dataset
    support_imgs = []
    for i in range(n_support):
        img = load_and_preprocess_image(support_paths[i], rot)
        support_imgs.append(tf.expand_dims(img, 0))
    ds_support = tf.concat(support_imgs, axis=0)

    # Create query dataset
    query_imgs = []
    for i in range(n_query):
        img = load_and_preprocess_image(query_paths[i], rot)
        query_imgs.append(tf.expand_dims(img, 0))
    ds_query = tf.concat(query_imgs, axis=0)

    return ds_support, ds_query


def load(config, splits):
    split_dir = os.path.join(OMNIGLOT_DATA_DIR, 'splits', config['data.split'])

    ret = {}
    for split in splits:
        # n_way (number of classes per episode)
        if split in ['val', 'test'] and config['data.test_way'] != 0:
            n_way = config['data.test_way']
        else:
            n_way = config['data.train_way']i

        # n_support (number of support examples per class)
        if split in ['val', 'test'] and config['data.test_n_support'] != 0:
            n_support = config['data.test_n_support']
        else:
            n_support = config['data.train_n_support']

        # n_query (number of query examples per class)
        if split in ['val', 'test'] and config['data.test_n_query'] != 0:
            n_query = config['data.test_n_query']
        else:
            n_query = config['data.train_n_query']

        # n_episodes (number of episodes per epoch)
        if split in ['val', 'test']:
            n_episodes = config['data.test_episodes']
        else:
            n_episodes = config['data.train_episodes']

        class_names = []
        with open(os.path.join(split_dir, f"{split}.txt"), 'r') as f:
            for class_name in f.readlines():
                class_names.append(class_name.rstrip('\n'))

        class_paths, rotates = class_names_to_paths(class_names)
        classes, img_paths, rotatess = get_class_images_paths(class_paths, rotates)

        class_paths_ds = tf.data.Dataset.from_tensor_slices(classes)
        img_paths_ds = tf.data.Dataset.from_tensor_slices(img_paths)
        rotates_ds =  tf.data.Dataset.from_tensor_slices(rotatess)

        class_paths_ds = tf.data.Dataset.zip((class_paths_ds, img_paths_ds, rotates_ds))
        class_imgs_ds = class_paths_ds.map(load_class_images)

        ret[split] = class_img_ds

    return ret
