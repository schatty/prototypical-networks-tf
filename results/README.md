## Results

Summary of conducted training procedures and corresponding results. Anyone who wants to contribute by conducting experiments are welcome to modify the table below.


### Omniglot

| Accuracy                    | 98.5%            | 94.1%            | 85.4%            | 96.3%            |
|-----------------------------|------------------|------------------|------------------|------------------|
| Author                      | Igor Kuznetsov   | Igor Kuznetsov   | Igor Kuznetsov   | Igor Kuznetsov   |
| data.split                  | vinyals          | vinyals          | vinyals          | vinyals          |
| data.train_way              | 60               | 60               | 60               | 60               |
| data.train_n_support        | 5                | 5                | 5                | 5                |
| data.train_n_query          | 5                | 5                | 5                | 5                |
| data.test_way (val)         | 5                | 5                | 5                | 5                |
| data.test_n_support (val)   | 5                | 5                | 5                | 5                |
| data.test_n_query (val)     | 15               | 15               | 15               | 15               |
| data.train_episodes         | 100              | 100              | 100              | 100              |
| model.x_dim                 | 28,28,1          | 28,28,1          | 28,28,1          | 28,28,1          |
| model.z_dim                 | 64               | 64               | 64               | 64               |
| train.epochs                | 500              | 500              | 500              | 500              |
| train.optim_method          | Adam             | Adam             | Adam             | Adam             |
| train.lr                    | 0.001            | 0.001            | 0.001            | 0.001            |
| train.patience              | 200              | 200              | 200              | 200              |
| data.test_way (test)        | 5                | 5                | 20               | 20               |
| data.test_n_support (test)  | 5                | 1                | 1                | 5                |
| data.test_n_query (test)    | 5                | 1                | 1                | 5                |
| data.test_n_episodes (test) | 1000             | 1000             | 1000             | 1000             |
| Encoder CNN architecture    | original (paper) | original (paper) | original (paper) | original (paper) |
| seed                        | 2019             | 2019             | 2019             | 2019             |

### MiniImagenet

| Accuracy                    | 56.6%            | 56.8%            | 36.6%            | 62.7%            | 62.6%            |
|-----------------------------|------------------|------------------|------------------|------------------|------------------|
| Author                      | Igor Kuznetsov   |                  | Igor Kuznetsov   | Igor Kuznetsov   | Igor Kuznetsov   |
| data.split                  | ravi             | ravi             | ravi             | ravi             | ravi             |
| data.train_way              | 20               | 20               | 20               | 20               | 20               |
| data.train_n_support        | 5                | 5                | 5                | 5                | 5                |
| data.train_n_query          | 5                | 5                | 5                | 5                | 5                |
| data.test_way (val)         | 5                | 5                | 5                | 5                | 5                |
| data.test_n_support (val)   | 5                | 5                | 5                | 5                | 5                |
| data.test_n_query (val)     | 15               | 15               | 15               | 15               | 15               |
| data.train_episodes         | 100              | 100              | 100              | 100              | 100              |
| model.x_dim                 | 84,84,3          | 84,84,3          | 84,84,3          | 84,84,3          | 84,84,3          |
| model.z_dim                 | 64               | 64               | 64               | 64               | 64               |
| train.epochs                | 500              | 500              | 500              | 500              | 500              |
| train.optim_method          | Adam             | Adam             | Adam             | Adam             | Adam             |
| train.lr                    | 0.001            | 0.001            | 0.001            | 0.001            | 0.001            |
| train.patience              | 200              | 200              | 200              | 200              | 200              |
| data.test_way (test)        | 5                | 5                | 5                | 5                | 5                |
| data.test_n_support (test)  | 5                | 5                | 1                | 10               | 10               |
| data.test_n_query (test)    | 5                | 1                | 1                | 10               | 1                |
| data.test_n_episodes (test) | 1000             | 1000             | 1000             | 1000             | 1000             |
| Encoder CNN architecture    | original (paper) | original (paper) | original (paper) | original (paper) | original (paper) |
| seed                        | 2019             | 2019             | 2019             | 2019             | 2019             |
