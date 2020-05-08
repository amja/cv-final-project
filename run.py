import numpy as np
import tensorflow as tf
from model import Model
from preprocess import Datasets
import hyperparameters as hp
from functools import partial
from our_tensorboard_utils import VisImageOutput

def train(model, dataset):
    checkpoint_path = "./your_model_checkpoints/"		
    load_checkpoint = None		

    callback_list = [		
        tf.keras.callbacks.ModelCheckpoint(		
        filepath=checkpoint_path + "weights.e{epoch:02d}-",		
        save_best_only=True,		
        save_weights_only=True),		
        tf.keras.callbacks.TensorBoard(		
        update_freq='batch',		
        profile_batch=0),
        VisImageOutput(dataset.train_data)		
    ]

    model.fit(
        x=dataset.train_data,
        validation_data=dataset.test_data,
        epochs=hp.num_epochs,
        callbacks=callback_list,
    )
def test(model, dataset):
    model.evaluate(
        x=dataset.test_data,
        verbose=1,
    )

def main():
    datasets = Datasets("data/imagenet")
    model = Model()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 1)))
    model.summary()

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn)
    
    train(model, datasets)
main()
