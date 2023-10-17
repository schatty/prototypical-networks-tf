# Prototypical Networks for Few-shot in TensorFlow 2.0
Implementation of Prototypical Networks for Few-shot Learning paper (https://arxiv.org/abs/1703.05175) in TensorFlow 2.0. Model has been tested on Omniglot and miniImagenet datasets with the same splitting as in the paper.

<img width="896" alt="Screenshot 2019-04-02 at 9 53 06 AM" src="https://user-images.githubusercontent.com/23639048/55438102-5d9e4c00-55a9-11e9-86e2-b4f79f880b83.png">

### Dependencies and Installation
* The code has been tested on Ubuntu 18.04 with Python 3.6.8 and TensorFlow 2.0.0-alpha0
* The two main dependencies are TensorFlow and Pillow package (Pillow is included in dependencies)
* To install `prototf` lib run `python setup.py install`
* Run `bash data/download_omniglot.sh` from repo's root directory to download Omniglot dataset
* miniImagenet was downloaded from brilliant repo from `renmengye` (https://github.com/renmengye/few-shot-ssl-public) and placed into `data/mini-imagenet` folder

### Repository Structure

The repository organized as follows. `data` directory contains scripts for dataset downloading and used as a default directory for datasets. `prototf` is the library containing the model itself (`prototf/models`) and logic for datasets loading and processing (`prototf/data`). `scripts` directory contains scripts for launching the training. `train/run_train.py` and `eval/run_eval.py` launch training and evaluation respectively. `tests` folder contains basic training procedure on small-valued parameters to check general correctness. `results` folder contains .md file with current configuration and details of conducted experiments.

### Training

* Training and evaluation configurations are specified through config files, each config describes single train+eval environment.
* Run `python scripts/train/run_train.py --config scripts/config_omniglot.conf` to run training on Omniglot with default parameters.
* Run `python scripts/train/run_train.py --config scripts/config_miniimagenet.conf` to run training on miniImagenet with default parameters

### Evaluating

* Run `python scripts/eval/run_eval.py --config scripts/config_omniglot.conf` to run evaluation on Omniglot
* Run `python scripts/eval/run_eval.py --config scripts/config_miniimagenet.conf` to run evaluation on miniImagenet

### Tests

* Run `python -m unittest tests/test_omniglot.py` from repo's root to test Omniglot
* Run `python -m unittest tests/test_mini_imagenet.py` from repo's root test miniImagenet 

### Results

Omniglot:

| Environment                 | 5-way-5-shot     | 5-way-1-shot     | 20-way-5-shot    | 20-way-1shot     |
|-----------------------------|------------------|------------------|------------------|------------------|
| Accuracy                    | 99.4%            | 97.4%            | 98.4%            | 92.2%            |

miniImagenet

| Environment                 | 5-way-5-shot     | 5-way-1-shot     | 
|-----------------------------|------------------|------------------|
| Accuracy                    | 66.0%            | 43.5%            |

Additional settings can be found in `results` folder in the root of repository. 

