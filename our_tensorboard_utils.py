import io
import os
import sklearn.metrics
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import hyperparameters as hp
# from preprocess as pp
import gc

class VisImageOutput(tf.keras.callbacks.Callback):
    def __init__(self, dataset):
        super(VisImageOutput, self).__init__()
        self.train_data = dataset.train_data
        self.dataset = dataset
        print("Done setting up image labeling logger.")
    
    # At the end of each epoch, visualize the current 
    # image output
    def on_epoch_end(self, epoch, logs=None):
        self.log_image_progress(epoch, logs)
    
    def log_image_progress(self, epoch_num, logs):
        # 224 x 224 x 1
        img = self.train_data.take(1).as_numpy_iterator().next()
        test_img_l = img[0][0]
        img3136by313 = tf.squeeze(self.model(tf.expand_dims(test_img_l, 0)))
        rgb_trained = self.dataset.model_output_to_tensorboard(test_img_l, img3136by313)
        rgb_truth = self.dataset.model_output_to_tensorboard(test_img_l, img[1][0])
        imgrank4 = tf.expand_dims(tf.concat([rgb_truth, rgb_trained], axis=1), 0)
        # Creates a file writer for the log directory.
        log_path = './logs/image_progress'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        file_writer = tf.summary.create_file_writer(log_path)
    
        # Using the file writer, log the reshaped image.
        with file_writer.as_default():
            tf.summary.image("Image Progress!", imgrank4, step=epoch_num)
        
        gc.collect()
