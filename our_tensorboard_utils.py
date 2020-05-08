import io
import os
import sklearn.metrics
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import hyperparameters as hp
import preprocess as pp

class VisImageOutput(tf.keras.callbacks.Callback):
    def __init__(self, datasets):
        super(VisImageOutput, self).__init__()
        self.datasets = datasets

        print("Done setting up image labeling logger.")
    
    # At the end of each epoch, visualize the current 
    # image output
    def on_epoch_end(self, epoch, logs=None):
        self.log_image_progress(epoch, logs)
    
    def log_image_progress(self, epoch_num, logs):
        batches = [batch for batch in self.train_data]
        # 224 x 224 x 1
        test_img_l = batches[0][0][0]
        img3136by313 = self.model(test_img_l)
        imgrank4 = pp.model_output_to_tensorboard(img3136by313)

        # Creates a file writer for the log directory.
        log_path = './logs/image_progress'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        file_writer = tf.summary.create_file_writer(log_path)
    
        # Using the file writer, log the reshaped image.
        with file_writer.as_default():
            tf.summary.image("Image Progress!", imgrank4, step=epoch_num)
