# Prototypical Networks for Few-shot in TensorFlow 2.0
Implementation of Prototypical Networks for Few-shot Learning paper (https://arxiv.org/abs/1703.05175) in TensorFlow 2.0. The implementation is based on official PyTorch version from the author (https://github.com/jakesnell/prototypical-networks). Model has been tested on Omniglot and miniImagenet datasets with the same splitting as in the paper.

<img width="896" alt="Screenshot 2019-04-02 at 9 53 06 AM" src="https://user-images.githubusercontent.com/23639048/55438102-5d9e4c00-55a9-11e9-86e2-b4f79f880b83.png">

### Dependencies and Installation
* The code has been tested on Ubuntu 18.04 with Python 3.6.8 and TensorFflow 2.0.0-alpha0
* The two main dependencies are TensorFlow and Pillow package (included in dependencies)
* To install `prototf` lib run `pytnon setup.py install`
* Run `bash data/download_omniglot.sh` from repo's root directory to download Omniglot dataset
* miniImagenet was downloaded from brilliant repo from `renmengye` (https://github.com/renmengye/few-shot-ssl-public) and placed into `data/mini-imagenet` folder

### Repository Structure

The repository organized as follows. `data` directory contains scripts for dataset downloading and used as default directory for datasets. `prototf` is the library containing the model itself (`prototf/models`) and logit for datasets processing (`prototf/data`). `scripts` directory contains scripts for launching the training. `train/setup_train.py` and `eval/setup_eval.py` contain all the configurations for training and evaluating respectively. `tests` contain basic training procedure on small-valued parameters to check that everything will be ok during training. `results` contains .md file with current results.

### Training

* Run `python scripts/train/run_train.py` to run training procedure on Omniglot with default parameters.
* Training on miniImagenet requried a bit different set of parameters. Training can be launched for example as `python scripts/train/run_train.py --model.x_dim '84,84,3' --data.cuda --data.dataset mini-imagenet --data.split ravi --data.train_way 20`

### Evaluating

* Run `python scripts/eval/run_eval.py` to run evaluation procedure on test set with default parameters on Omniglot.
* Run `python scripts/eval/run_eval.py --model.x_dim '84,84,3' --train.model_path model_miniimagenet.h5 --data.cuda --data.dataset mini-imagenet --data.split ravi --data.test_way 20` to evaluate miniImagenet

### Tests

* Run `python -m unittest tests/test_omniglot.py` from repo's root to test Omniglot
* Run `python -m unittest tests/test_mini_imagenet.py` from repo's root test miniImagenet 

### Experiment results and future work

Conducted experiments uploaded to the `results` folder in the root of repository. Opening of Issues and getting in touch for implemenation details are welcome (:

