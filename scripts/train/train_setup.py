from prototf.data import loader


class Experiment(object):
    def __init__(self):
        pass

    def train(self, **kwargs):
        pass


def train(config):
    print("Config: ", config)

    train_loader = loader.load(config, ['train'])
    print("Train loader: ", type(train_loader))

    experiment = Experiment()
    experiment.train(
        model=None,
        loader=train_loader,
        optim_method=config['train.optim_method'],
        optim_config={'lr': config['train.lr'],
                      'weight_decay': config['train.weight_decay']},
        max_epoch=config['train.epochs']
    )