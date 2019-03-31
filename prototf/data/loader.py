from .omniglot import load_omniglot


def load(data_dir, config, splits):
    if config['data.dataset'] == "omniglot":
        ds = load_omniglot(data_dir, config, splits)
    else:
        raise ValueError(f"Unknow dataset: {config['data.dataset']}")
    return ds