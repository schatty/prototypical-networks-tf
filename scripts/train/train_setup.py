from prototf.data import load


class Experiment(object):
    def __init__(self):
        pass

    def train(self, loader, **kwargs):
        for img in loader:
            print(img[0].shape)
            break
        print("Success!")


def train(config):
    print("Config: ", config)
    data_dir = './data/omniglot'
    ret = load(data_dir, config, ['train'])
    img_ds = ret['train']

    experiment = Experiment()
    experiment.train(
        model=None,
        loader=img_ds,
        optim_method=config['train.optim_method'],
        optim_config={'lr': config['train.lr'],
                      'weight_decay': config['train.weight_decay']},
        max_epoch=config['train.epochs']
    )