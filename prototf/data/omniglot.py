import os
import glob
from functools import partial
import numpy as np
import tensorflow as tf
from PIL import Image


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


def load_and_preprocess_image(img_path, rot):
    """
    Load and return preprocessed image.

    Args:
        img_path (str): path to the image on disk.

    Returns (Tensor): preprocessed image

    """
    img = Image.open(img_path).resize((28, 28)).rotate(rot)
    img = np.asarray(img)
    img = 1 - img
    return np.expand_dims(img, -1)


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

        data = np.zeros([len(classes), len(img_paths[0]), w, h, c])
        for i_class in range(len(classes)):
            for i_img  in range(len(img_paths[i_class])):
                data[i_class, i_img, :, :, :] = load_and_preprocess_image(img_paths[i_class][i_img], rotates[i_class])

        data = tf.constant(data, dtype=tf.float16)
        img_ds = tf.data.Dataset.from_tensor_slices(np.arange(n_episodes))
        img_ds = img_ds.map(partial(load_episode, data=data, n_way=n_way, n_support=n_support, n_query=n_query))
        ret[split] = img_ds
    return ret



if __name__ == "__main__":
    config = {
        'data.train_way': 13,
        'data.test_way': 13,
        'data.train_n_query': 5,
        'data.train_n_support': 5,
        'data.test_n_query': 5,
        'data.test_n_support': 5,
        'data.train_episodes': 10,
        'data.test_episodes': 10,
        'data.split': 'vinyals',
        'model.x_dim': '28,28,1'
    }
    load_omniglot('data/omniglot', config, ['test'])
