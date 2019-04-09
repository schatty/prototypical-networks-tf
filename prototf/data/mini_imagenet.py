import os
import numpy as np
import pickle
from PIL import Image
from functools import partial
import tensorflow as tf

class DataLoader(object):
    def __init__(self, data, n_classes, n_way, n_support, n_query):
        self.data = data
        self.n_way = n_way
        self.n_classes = n_classes
        self.n_support = n_support
        self.n_query = n_query

    def get_next_episode(self):
        n_examples = self.data.shape[1]
        support = np.zeros([self.n_way, self.n_support, 84, 84, 3], dtype=np.float32)
        query = np.zeros([self.n_way, self.n_query, 84, 84, 3], dtype=np.float32)
        classes_ep = np.random.permutation(self.n_classes)[:self.n_way]

        for i, i_class in enumerate(classes_ep):
            selected = np.random.permutation(n_examples)[:self.n_support + self.n_query]
            support[i] = self.data[i_class, selected[:self.n_support]]
            query[i] = self.data[i_class, selected[self.n_support:]]

        return support, query


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
            n_support = config['data.test_support']
        else:
            n_support = config['data.train_support']

        # n_query (number of query examples per class)
        if split in ['val', 'test']:
            n_query = config['data.test_query']
        else:
            n_query = config['data.train_query']

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
        data /= 255.
        data[:, :, :, 0] = (data[:, :, :, 0] - 0.485) / 0.229
        data[:, :, :, 1] = (data[:, :, :, 1] - 0.456) / 0.224
        data[:, :, :, 2] = (data[:, :, :, 2] - 0.406) / 0.225

        data_loader = DataLoader(data, data.shape[0], n_way, n_support, n_query)
        ret[split] = data_loader

    return ret
