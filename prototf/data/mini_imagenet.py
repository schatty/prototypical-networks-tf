import os
import numpy as np
import pickle
from PIL import Image
from functools import partial
import tensorflow as tf


def dump_on_disk(data_dir, split, images):
    """
    Dump miniImagenet from np array to the .png files on disk.

    Args:
        data_dir (str): path to the directory where to dump
        split (str): 'train'|'val'|'test'
        images (np.ndarray): numpy representation of images

    Returns: None

    """
    split_path = os.path.join(data_dir, 'data', split)
    if not os.path.exists(split_path):
        os.makedirs(split_path)
    for i, img in enumerate(images):
        img_pillow = Image.fromarray(img)
        img_pillow.save(f"{split_path}/{i}.png", "PNG")


def preprocess_image(img):
    """
    Preprocess single image.

    Args:
        img (Tensor of type string): image readed by tensorflow

    Returns (Tensor of type tf.float32):

    """
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (84, 84))
    img /= 255.0
    return img


def load_and_preprocess_image(img_path):
    """
    Load and return preprocessed image.

    Args:
        img_path (str): path to the image on disk.

    Returns (Tensor): preprocessed image

    """
    image = tf.io.read_file(img_path)
    return preprocess_image(image)


def load_class_images(n_support, n_query, img_paths):
    """
    Load support and query datasets with processed images.

    Args:
        images (np.ndarray): numpy array of images
        n_support (int): number of support samples
        n_query (int): number of query samples
        class_names (str): name of the class
        img_inds (list): indices of the images belonging to the class

    Returns (tf.data.Dataset, tf.data.Dataset): support and query datasets
    
    """
    n_examples = img_paths.shape[0]
    example_inds = tf.range(n_examples)
    example_inds = tf.random.shuffle(example_inds)

    # Get indidces for support and query datasets
    support_inds = example_inds[:n_support]
    support_paths = tf.gather(img_paths, support_inds)
    query_inds = example_inds[n_support:]
    query_paths = tf.gather(img_paths, query_inds)

    # Support dataset
    support_imgs_proc = []
    for i in range(n_support):
        img_proc = load_and_preprocess_image(support_paths[i])
        img_proc = tf.expand_dims(img_proc, 0)
        support_imgs_proc.append(img_proc)
        support_imgs = tf.concat(support_imgs_proc, axis=0)

    # Query dataset
    query_imgs_proc = []
    for i in range(n_support):
        img_proc = load_and_preprocess_image(query_paths[i])
        img_proc = tf.expand_dims(img_proc, 0)
        query_imgs_proc.append(img_proc)
        query_imgs = tf.concat(query_imgs_proc, axis=0)

    return support_imgs, query_imgs

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
            dataset = pickle.load(f)
        images = [dataset['image_data'][i, :] for i in range(len(dataset['image_data']))]
        
        split_path = os.path.join(data_dir, 'data', split)
        if not os.path.exists(split_path):
            print(f"Dumping on {split} disk")
            dump_on_disk(data_dir, split, images)
        else:
            print("Path existing")
        
        data = np.array([[f"{split_path}/{v}.png" for v in dataset['class_dict'][k]] 
                                     for k in dataset['class_dict']])
        image_paths_ds = tf.data.Dataset.from_tensor_slices(data)
        img_ds = image_paths_ds.map(partial(load_class_images, n_support, n_query))
        img_ds = img_ds.shuffle(len(dataset['class_dict']))
        img_ds = img_ds.repeat()
        img_ds = img_ds.batch(n_way)
        ret[split] = img_ds

    return ret
