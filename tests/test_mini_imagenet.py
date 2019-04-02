import os
import unittest

from scripts import train


class TestsMiniImagenet(unittest.TestCase):

    def test_1_shot_1_way(self):
        config = {
            "data.dataset": "mini-imagenet",
            "data.split": "ravi",
            "data.train_way": 1,
            "data.train_n_support": 1,
            "data.train_n_query": 1,
            "data.test_way": 1,
            "data.test_n_support": 1,
            "data.test_n_query": 1,
            "data.train_episodes": 10,
            "data.cuda": False,
            "data.gpu": 0,
            "model.x_dim": "84,84,1",
            "model.z_dim": 64,
            "train.epochs": 2,
            'train.optim_method': "Adam",
            "train.lr": 0.001,
            "train.patience": 5,
            "train.model_path": 'test_omniglot.h5'
        }
        train(config)
        os.remove('test_omniglot.h5')

    def test_5_shot_5_way(self):
        config = {
            "data.dataset": "mini-imagenet",
            "data.split": "ravi",
            "data.train_way": 5,
            "data.train_n_support": 5,
            "data.train_n_query": 5,
            "data.test_way": 5,
            "data.test_n_support": 5,
            "data.test_n_query": 5,
            "data.train_episodes": 10,
            "data.cuda": False,
            "data.gpu": 0,
            "model.x_dim": "84,84,1",
            "model.z_dim": 64,
            "train.epochs": 2,
            'train.optim_method': "Adam",
            "train.lr": 0.001,
            "train.patience": 5,
            "train.model_path": 'test_mi_net.h5'
        }
        train(config)
        os.remove('test_mi_net.h5')

    def test_10_shot_1_way(self):
        config = {
            "data.dataset": "mini-imagenet",
            "data.split": "ravi",
            "data.train_way": 1,
            "data.train_n_support": 10,
            "data.train_n_query": 10,
            "data.test_way": 1,
            "data.test_n_support": 10,
            "data.test_n_query": 10,
            "data.train_episodes": 10,
            "data.cuda": False,
            "data.gpu": 0,
            "model.x_dim": "84,84,1",
            "model.z_dim": 64,
            "train.epochs": 2,
            'train.optim_method': "Adam",
            "train.lr": 0.001,
            "train.patience": 5,
            "train.model_path": 'test_mi_net.h5'
        }
        train(config)
        os.remove('test_mi_net.h5')

    def test_1_shot_50_way(self):
        config = {
            "data.dataset": "mini-imagenet",
            "data.split": "ravi",
            "data.train_way": 50,
            "data.train_n_support": 1,
            "data.train_n_query": 1,
            "data.test_way": 50,
            "data.test_n_support": 1,
            "data.test_n_query": 1,
            "data.train_episodes": 10,
            "data.cuda": False,
            "data.gpu": 0,
            "model.x_dim": "84,84,1",
            "model.z_dim": 64,
            "train.epochs": 2,
            'train.optim_method': "Adam",
            "train.lr": 0.001,
            "train.patience": 5,
            "train.model_path": 'test_mi_net.h5'
        }
        train(config)
        os.remove('test_mi_net.h5')

