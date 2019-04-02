import argparse

from eval_setup import eval

parser = argparse.ArgumentParser(description='Run evaluation')

default_ds = "omniglot"
default_split = "vinyals"
default_test_way = 50
default_test_n_support = 5
default_n_query = 5
default_test_n_query = default_n_query
default_test_episodes = 10

# data
parser.add_argument("--data.dataset", type=str, default=default_ds,
                    help=f"dataset name (default: {default_ds}")
parser.add_argument("--data.split", type=str, default=default_split,
                    help=f"splitting name (default: {default_split}")
parser.add_argument("--data.test_way", type=int, default=default_test_way,
                    help=f"number of classes per episode in test.")
parser.add_argument("--data.test_n_support", type=int, default=default_test_n_support,
                    help=f"number of support examples per class in test")
parser.add_argument("--data.test_n_query", type=int, default=default_test_n_query,
                    help=f"number of query examples per class in test")
parser.add_argument("--data.test_episodes", type=int, default=default_test_episodes,
                    help=f"number of test episodes per epoch (default: {default_test_episodes}")
parser.add_argument("--data.cuda", action='store_true',
                    help=f"Train on GPU (default: False)")

# model
default_input = "28,28,1"
parser.add_argument("--model.x_dim", type=str, default=default_input,
                    help=f"dimensionality of input shapes (default: {default_input})")
parser.add_argument("--train.model_path", type=str, default="./model.h5",
                    help="Path to the saved model (default: ./model.h5)")

# Run evaluation
args = vars(parser.parse_args())
eval(args)
