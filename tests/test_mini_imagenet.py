import os
import unittest

from scripts import train


class TestsMiniImagenet(unittest.TestCase):

    def test_1_shot_1_way(self):
        config = {
            "data.dataset": "mini-imagenet",
            "data.split": "ravi",
            "data.train_way": 1,
            "data.train_support": 1,
            "data.train_query": 1,
            "data.test_way": 1,
            "data.test_support": 1,
            "data.test_query": 1,
            "data.episodes": 10,
            "data.cuda": False,
            "data.gpu": 0,
            "model.x_dim": "84,84,3",
            "model.z_dim": 64,
            "train.epochs": 1,
            'train.optim_method': "Adam",
            "train.lr": 0.001,
            "train.patience": 5,
            "model.save_path": 'test_miniimagenet.h5'
        }
        train(config)
        os.remove('test_miniimagenet.h5')

    def test_5_shot_5_way(self):
        config = {
            "data.dataset": "mini-imagenet",
            "data.split": "ravi",
            "data.train_way": 5,
            "data.train_support": 5,
            "data.train_query": 5,
            "data.test_way": 5,
            "data.test_support": 5,
            "data.test_query": 5,
            "data.episodes": 3,
            "data.cuda": False,
            "data.gpu": 0,
            "model.x_dim": "84,84,3",
            "model.z_dim": 64,
            "train.epochs": 1,
            'train.optim_method': "Adam",
            "train.lr": 0.001,
            "train.patience": 5,
            "model.save_path": 'test_mi_net.h5'
        }
        train(config)
        os.remove('test_mi_net.h5')

    def test_10_shot_1_way(self):
        config = {
            "data.dataset": "mini-imagenet",
            "data.split": "ravi",
            "data.train_way": 1,
            "data.train_support": 10,
            "data.train_query": 10,
            "data.test_way": 1,
            "data.test_support": 10,
            "data.test_query": 10,
            "data.episodes": 3,
            "data.cuda": False,
            "data.gpu": 0,
            "model.x_dim": "84,84,3",
            "model.z_dim": 64,
            "train.epochs": 1,
            'train.optim_method': "Adam",
            "train.lr": 0.001,
            "train.patience": 5,
            "model.save_path": 'test_mi_net.h5'
        }
        train(config)
        os.remove('test_mi_net.h5')

