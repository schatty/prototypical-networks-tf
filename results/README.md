## Results

Summary of conducted training procedures and corresponding results. Anyone who wants to contribute by conducting experiments are welcome to modify the table below.


### Omniglot

| Accuracy                    | 99.5%            | 97.4%            | 92.2%            | 98.4%            |
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
| train.epochs                | 100              | 100              | 100              | 100              |
| train.optim_method          | Adam             | Adam             | Adam             | Adam             |
| train.lr                    | 0.001            | 0.001            | 0.001            | 0.001            |
| train.patience              | 100              | 100              | 100              | 100              |
| data.test_way (test)        | 5                | 5                | 20               | 20               |
| data.test_n_support (test)  | 5                | 1                | 1                | 5                |
| data.test_n_query (test)    | 5                | 1                | 1                | 5                |
| data.test_n_episodes (test) | 1000             | 1000             | 1000             | 1000             |
| Encoder CNN architecture    | original (paper) | original (paper) | original (paper) | original (paper) |
| seed                        | 2019             | 2019             | 2019             | 2019             |

### MiniImagenet

| Accuracy                    | 66%              | 43.5%            |
|-----------------------------|------------------|------------------|
| Author                      | Igor Kuznetsov   | Igor Kuznetsov   |
| data.split                  | ravi             | ravi             |
| data.train_way              | 30               | 30               |
| data.train_n_support        | 5                | 5                |
| data.train_n_query          | 15               | 15               |
| data.test_way (val)         | 5                | 5                |
| data.test_n_support (val)   | 5                | 5                |
| data.test_n_query (val)     | 15               | 15               |
| data.train_episodes         | 100              | 100              |
| model.x_dim                 | 84,84,3          | 84,84,3          |
| model.z_dim                 | 64               | 64               |
| train.epochs                | 300              | 300              |
| train.optim_method          | Adam             | Adam             | 
| train.lr                    | 0.001            | 0.001            |
| train.patience              | 100              | 100              |
| data.test_way (test)        | 5                | 5                |
| data.test_n_support (test)  | 5                | 1                |
| data.test_n_query (test)    | 5                | 1                |
| data.test_n_episodes (test) | 1000             | 1000             |
| Encoder CNN architecture    | original (paper) | original (paper) |
| seed                        | 2019             | 2019             |
