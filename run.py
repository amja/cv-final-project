import numpy as np
import tensorflow as tf
from model import Model
from preprocess import Datasets
import hyperparameters as hp
from functools import partial
# from tensorboard_utils import ImageLabelingLogger
import os

def train(model, dataset):
    # taken from Project 4
    checkpoint_path = "./your_model_checkpoints/"
    load_checkpoint = None

    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path + \
                    "weights.e{epoch:02d}-",
            save_best_only=True,
            save_weights_only=True),
        tf.keras.callbacks.TensorBoard(
            update_freq='batch',
            profile_batch=0),
    ]
    model.fit(
        x=dataset.train_data,
        validation_data=dataset.test_data,
        epochs=hp.num_epochs,
        steps_per_epoch=hp.steps_per_epoch,
        # batch_size=hp.batch_size,
        callbacks=[],
    )
    if load_checkpoint is not None:
        model.load_weights(load_checkpoint)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)


def test(model, dataset):
    model.evaluate(
        x=dataset.test_data,
        verbose=1,
    )

def main():
    datasets = Datasets("data")
    model = Model()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 1)))
    model.summary()

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn)
    
    train(model, datasets)
main()
