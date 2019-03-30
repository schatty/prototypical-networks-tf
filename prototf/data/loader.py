from .omniglot import load_omniglot


def load(config, splits):
    if config['data.dataset'] == "omniglot":
        ds = load_omniglot(config, splits)
    else:
        raise ValueError(f"Unknow dataset: {config['data.dataset']}")
    return ds