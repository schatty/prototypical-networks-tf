from .omniglot import load_omniglot
from .mini_imagenet import load_mini_imagenet


def load(data_dir, config, splits):
    """
    Load specific dataset.

    Args:
        data_dir (str): path to the dataset directory.
        config (dict): general dict with settings.
        splits (list): list of strings 'train'|'val'|'test'.

    Returns (dict): dictionary with keys 'train'|'val'|'test'| and values
    as tensorflow Dataset objects.

    """
    if config['data.dataset'] == "omniglot":
        ds = load_omniglot(data_dir, config, splits)
    elif config['data.dataset'] == "mini-imagenet":
        ds = load_mini_imagenet(data_dir, config, splits)
    else:
        raise ValueError(f"Unknow dataset: {config['data.dataset']}")
    return ds
