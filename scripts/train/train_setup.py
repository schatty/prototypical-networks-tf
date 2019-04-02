import numpy as np
import tensorflow as tf

from prototf.models import Prototypical
from prototf.data import load
from prototf import TrainEngine


def train(config):
    print("Config: ", config)
    #data_dir = '/home/igor/dl/prototypical-networks/data/omniglot'
    data_dir = '/Users/sirius/code/prototypical-networks-tf/data/omniglot'
    ret = load(data_dir, config, ['train', 'val'])
    train_loader = ret['train']
    val_loader = ret['val']

    # Determine device
    if config['data.cuda']:
        device_name = 'GPU:0'
    else:
        device_name = 'CPU:0'

    # Setup training operations
    model = Prototypical()
    optimizer = tf.keras.optimizers.Adam(config['train.lr'])

    train_loss = tf.metrics.Mean(name='train_loss')
    val_loss = tf.metrics.Mean(name='val_loss')
    train_acc = tf.metrics.Mean(name='train_accuracy')
    val_acc = tf.metrics.Mean(name='val_accuracy')
    val_losses = []

    def loss(support, query):
        loss, acc = model(support, query)
        return loss, acc

    @tf.function
    def train_step(loss_func, support, query):
        # Forward & update gradients
        with tf.GradientTape() as tape:
            loss, acc = loss_func(support, query)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))

        # Log loss and accuracy for step
        train_loss(loss)
        train_acc(acc)

    @tf.function
    def val_step(loss_func, support, query):
        loss, acc = loss_func(support, query)
        val_loss(loss)
        val_acc(acc)

    train_engine = TrainEngine()

    # Set hooks on training process
    def on_start(state):
        print("Training started.")
    train_engine.hooks['on_start'] = on_start

    def on_end(state):
        print("Training ended.")
    train_engine.hooks['on_end'] = on_end

    def on_start_epoch(state):
        print(f"Epoch {state['epoch']} started.")
    train_engine.hooks['on_start_epoch'] = on_start_epoch

    def on_end_epoch(state):
        print(f"Epoch {state['epoch']} ended.")
        epoch = state['epoch']
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
        print(
            template.format(epoch + 1, train_loss.result(), train_acc.result(),
                            val_loss.result(),
                            val_acc.result() * 100))

        cur_loss = val_loss.result().numpy()
        if cur_loss < state['best_val_loss']:
            print("Saving new best model with loss: ", cur_loss)
            state['best_val_loss'] = cur_loss
            model.save(config['train.save_path'])
        val_losses.append(cur_loss)

        # Early stopping
        patience = config['train.patience']
        if len(val_losses) > patience \
                and max(val_loss[-patience:]) == val_losses[-1]:
            state['early_stopping_triggered'] = True
    train_engine.hooks['on_end_epoch'] = on_end_epoch

    def on_start_episode(state):
        print(f"Episode {state['total_episode']}")
        support, query = state['sample']
        loss_func = state['loss_func']
        train_step(loss_func, support, query)
    train_engine.hooks['on_start_episode'] = on_start_episode

    def on_end_episode(state):
        # Validation
        val_loader = state['val_loader']
        loss_func = state['loss_func']
        for support, query in val_loader:
            val_step(loss_func, support, query)

    train_engine.hooks['on_end_episode'] = on_end_episode

    with tf.device(device_name):
        train_engine.train(
            loss_func=loss,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['train.epochs'],
            n_episodes=config['data.train_episodes'])
