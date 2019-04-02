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


def load_class_images(n_support, n_query, img_paths, rot):
    """
    Given paths to the images of class, build support and query tf.Datasets.

    Args:
        n_support (int): number of images per support.
        n_query (int): number of images per query.
        img_paths (list): list of paths to the images for class.
        rot (int): rotation angle in degrees.

    Returns (tf.Dataset, tf.Dataset): support and query datasets.

    """
    # Shuffle indices of given images
    n_examples = img_paths.shape[0]
    example_inds = tf.range(n_examples)
    example_inds = tf.random.shuffle(example_inds)

    # Get indidces for support and query datasets
    support_inds = example_inds[:n_support]
    support_paths = tf.gather(img_paths, support_inds)
    query_inds = example_inds[n_support:]
    query_paths = tf.gather(img_paths, query_inds)

    # Build support dataset
    support_imgs = []
    for i in range(n_support):
        img = load_and_preprocess_image(support_paths[i])
        support_imgs.append(tf.expand_dims(img, 0))
    ds_support = tf.concat(support_imgs, axis=0)
    # Rotate dataset by given angle
    ds_support = tf.image.rot90(ds_support, k=rot//90)

    # Create query dataset
    query_imgs = []
    for i in range(n_query):
        img = load_and_preprocess_image(query_paths[i])
        query_imgs.append(tf.expand_dims(img, 0))
    ds_query = tf.concat(query_imgs, axis=0)
    # Rotate dataset by given angle
    ds_query = tf.image.rot90(ds_query, k=rot//90)

    return ds_support, ds_query


def load_omniglot(data_dir, config, splits):
    """
    Load omniglot dataset.

    Args:
        data_dir (str): path of the directory with 'splits', 'data' subdirs.
        config (dict): general dict with program settings.
        splits (list): list of strings 'train'|'val'|'test'

    Returns (dict): dictionary with keys as splits and values as tf.Dataset

    """
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

        # Get all class names
        class_names = []
        with open(os.path.join(split_dir, f"{split}.txt"), 'r') as f:
            for class_name in f.readlines():
                class_names.append(class_name.rstrip('\n'))

        # Get class names, images paths and rotation angles per each class
        class_paths, rotates = class_names_to_paths(data_dir, class_names)
        classes, img_paths, rotatess = get_class_images_paths(class_paths,
                                                              rotates)

        # Construct tf datasets for class names, img paths and rotates
        img_paths_ds = tf.data.Dataset.from_tensor_slices(img_paths)
        rotates_ds = tf.data.Dataset.from_tensor_slices(rotatess)
        class_paths_ds = tf.data.Dataset.zip((img_paths_ds, rotates_ds))
        # Convert three datasets above into one that outputs images of class
        class_imgs_ds = class_paths_ds.map(partial(load_class_images,
                                                   n_support, n_query))
        # Batch size is equal to way (number of classes per episode)
        class_imgs_ds = class_imgs_ds.batch(n_way)

        ret[split] = class_imgs_ds
    return ret
