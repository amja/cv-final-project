import numpy as np
import tensorflow as tf
from model import Model
from preprocess import Datasets
import hyperparameters as hp

def train(model, dataset):
    model.fit(
        x=dataset.train_data,
        validation_data=dataset.test_data,
        epochs=hp.num_epochs,
        callbacks=[],
    )
def test(model, dataset):
    model.evaluate(
        x=dataset.test_data,
        verbose=1,
    )

def main():
    datasets = Datasets("data")
    model = Model()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    model.summary()

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["accuracy"])
    
    train(model, datasets)
main()
