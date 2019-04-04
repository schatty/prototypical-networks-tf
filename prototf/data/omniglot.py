import os
import glob
from functools import partial
import numpy as np
import tensorflow as tf


def class_names_to_paths(data_dir, class_names):
    """
    Return full paths to the directories containing classes of images.

    Args:
        data_dir (str): directory with dataset
        class_names (list): names of the classes in format alphabet/name/rotate

    Returns (list, list): list of paths to the classes,
    list of stings of rotations codes
    """
    d = []
    rots = []
    for class_name in class_names:
        alphabet, character, rot = class_name.split('/')
        image_dir = os.path.join(data_dir, 'data', alphabet, character)
        d.append(image_dir)
        rots.append(rot)
    return d, rots


def get_class_images_paths(dir_paths, rotates):
    """
    Return class names, paths to the corresponding images and rotations from
    the path of the classes' directories.

    Args:
        dir_paths (list): list of the class directories
        rotates (list): list of stings of rotation codes.

    Returns (list, list, list): list of class names, list of lists of paths to
    the images, list of rotation angles (0..240) as integers.

    """
    classes, img_paths, rotates_list = [], [], []
    for dir_path, rotate in zip(dir_paths, rotates):
        class_images = sorted(glob.glob(os.path.join(dir_path, '*.png')))

        classes.append(dir_path)
        img_paths.append(class_images)
        rotates_list.append(int(rotate[3:]))
    return classes, img_paths, rotates_list


def preprocess_image(img):
    """
    Preprocess single image.

    Args:
        img (Tensor of type string): image readed by tensorflow

    Returns (Tensor of type tf.float32):

    """
    image = tf.image.decode_png(img, channels=1)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (28, 28))
    image /= 255.0
    image = 1-image
    return image


def load_and_preprocess_image(img_path):
    """
    Load and return preprocessed image.

    Args:
        img_path (str): path to the image on disk.

    Returns (Tensor): preprocessed image

    """
    image = tf.io.read_file(img_path)
    return preprocess_image(image)


def gen_episode(class_inds, img_paths, rotates, n_way, n_episodes,
                n_support, n_query):
    for i in range(n_episodes):
        np.random.shuffle(class_inds)
        class_way_inds = class_inds[:n_way]

        way_support_shots = []
        way_query_shots = []
        # Build support dataset
        for i_class in class_way_inds:
            img_class_paths = img_paths[i_class]
            img_inds = np.arange(len(img_class_paths))
            np.random.shuffle(img_inds)
            support_inds = img_inds[:n_support]
            query_inds = img_inds[n_support:n_support + n_query]

            support_imgs = []
            for i in support_inds:
                img = load_and_preprocess_image(img_class_paths[i])
                support_imgs.append(tf.expand_dims(img, 0))
            ds_support = tf.concat(support_imgs, axis=0)
            # Rotate dataset by given angle
            ds_support = tf.image.rot90(ds_support,
                                        k=rotates[i_class] // 90)
            way_support_shots.append(tf.expand_dims(ds_support, 0))

            # Query dataset
            query_imgs = []
            for i in query_inds:
                img = load_and_preprocess_image(img_class_paths[i])
                query_imgs.append(tf.expand_dims(img, 0))
            ds_query = tf.concat(query_imgs, axis=0)
            # Rotate dataset by given angle
            ds_query = tf.image.rot90(ds_query,
                                      k=rotates[i_class] // 90)
            way_query_shots.append(tf.expand_dims(ds_query, 0))

        way_support_shots = tf.concat(way_support_shots, axis=0)
        way_query_shots = tf.concat(way_query_shots, axis=0)

        yield way_support_shots, way_query_shots


def load_omniglot(data_dir, config, splits):
    """
    Load omniglot dataset.

    Args:
        data_dir (str): path of the directory with 'splits', 'data' subdirs.
        config (dict): general dict with program settings.
        splits (list): list of strings 'train'|'val'|'test'

    Returns (dict): dictionary with keys as splits and values as tf.Dataset

    """
    w, h, c = list(map(int, config['model.x_dim'].split(',')))
    split_dir = os.path.join(data_dir, 'splits', config['data.split'])
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

        # n_query (number of episodes per epoch)
        if split in ['val', 'test']:
            n_episodes = config['data.test_episodes']
        else:
            n_episodes = config['data.train_episodes']

        # Get all class names
        class_names = []
        with open(os.path.join(split_dir, f"{split}.txt"), 'r') as f:
            for class_name in f.readlines():
                class_names.append(class_name.rstrip('\n'))

        # Get class names, images paths and rotation angles per each class
        class_paths, rotates = class_names_to_paths(data_dir, class_names)
        classes, img_paths, rotates = get_class_images_paths(class_paths,
                                                              rotates)

        class_inds = np.arange(len(class_names))
        imgs_ds = tf.data.Dataset.from_generator(partial(gen_episode,
                                                         class_inds,
                                                         img_paths,
                                                         rotates,
                                                         n_way,
                                                         n_episodes,
                                                         n_support,
                                                         n_query),
                                                 (tf.float32, tf.float32),
                                                 (tf.TensorShape([n_way, n_support, w, h, c]),
                                                  tf.TensorShape([n_way, n_query, w, h, c])))
        ret[split] = imgs_ds
    return ret
