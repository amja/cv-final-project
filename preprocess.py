import os
import random
import numpy as np
import tensorflow as tf
import hyperparameters as hp
from lab_funcs import rgb_to_lab
from sklearn.cluster import KMeans, MiniBatchKMeans
import pickle

AUTOTUNE = tf.data.experimental.AUTOTUNE
class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path):
        self.train_path = os.path.join(data_path, "train")
        self.test_path = os.path.join(data_path, "test")

        # Mean and std for standardization
        self.mean = np.zeros((3,))
        self.std = np.ones((3,))
        self.calc_mean_and_std()
        self.get_q_probabilities()
        # Setup data generators
        self.train_data = self.get_data(self.train_path, True)
        self.test_data = self.get_data(self.test_path, False)

    '''
    This is the first step in getting a loss function that accounts for color rarity.
    We will have to reweight each pixel based on pixel color rarity. 
    This function gets the probabilities for each of 313 Q colors.
    Q colors are the quantized ab values which are "in-gamut" for RGB --
    (Values are [-110, 110] for CIELAB. For a given L, we break the 
    CIELAB space into boxes of different colors to make the task 
    easier (i.e. smaller than 220*220 possible colors).
    Only some of these are visible in RGB space, so we have 313 left.

    '''
    def get_q_probabilities(self):
        # Get list of all images in training directory
        file_list = []
        for root, _, files in os.walk(self.train_path):
            for name in files:
                file_list.append(os.path.join(root, name))

        # Shuffle filepaths
        random.shuffle(file_list)

        # Take sample of file paths
        file_list = file_list[:hp.preprocess_sample_size]

        # Allocate space in memory for Nx2xXxY ab images
        data_sample = np.zeros(
            (hp.preprocess_sample_size, hp.img_size, hp.img_size, 2))

        # Import images
        for i, file_path in enumerate(file_list):
            img = tf.io.read_file(file_path)
            # img now in LAB
            img = self.convert_img(img, False)
            # get img to just Ab
            ab_image = img[:,:,1:]
            data_sample[i] = ab_image
        
        # Reshape to get a list of ab colors 
        ab_colors_arr = data_sample.reshape((-1,2))
        
        if not os.path.isfile('kmeans_qcolors.pkl'):
            kmeans = self.gen_q_cc(ab_colors_arr)
        else:
            kmeans = pickle.load(open("kmeans_qcolors.pkl", "rb"))

        # Get Q colors for all training images
        training_ims_q_colors = kmeans.predict(ab_colors_arr[:1000])

        # tensor_images = tf.convert_to_tensor(data_sample, dtype=tf.float32)
        # quant_vals = tf.quantization.quantize(tensor_images, min_range=-110, max_range=110,T=tf.qint8)
        
        # Get probability dist for each Q color (v)
        v = training_ims_q_colors
        
        return v

    ''' From available images generate 313 cluster centers of ab colors'''
    def gen_q_cc(self, ab_colors):
        print('Generating q colors through kmeans!')
        kmeans = MiniBatchKMeans(n_clusters=313, init_size=313, max_iter=20).fit(ab_colors[:1000])
        pickle.dump(kmeans, open("kmeans_qcolors.pkl", "wb"))
        print('...Done.')
        return kmeans

    def get_img_q_color_from_ab(self, ab_img):
        kmeans = pickle.load(open("kmeans_qcolors.pkl", "rb"))
        
        q_colors = kmeans.predict(ab_img)
        return ab_img
    
    def calc_mean_and_std(self):
        # Get list of all images in training directory
        file_list = []
        for root, _, files in os.walk(self.train_path):
            for name in files:
                if name.endswith("jpg") or name.endswith("jpeg") or name.endswith("png"):
                    file_list.append(os.path.join(root, name))

        # Shuffle filepaths
        random.shuffle(file_list)

        # Take sample of file paths
        file_list = file_list[:hp.preprocess_sample_size]

        # Allocate space in memory for images
        data_sample = np.zeros(
            (hp.preprocess_sample_size, hp.img_size, hp.img_size, 3))

        # Import images
        for i, file_path in enumerate(file_list):
            img = tf.io.read_file(file_path)
            img = self.convert_img(img, False)
            data_sample[i] = img

        self.mean = np.mean(data_sample, axis=(0,1,2))
        self.std = np.std(data_sample, axis=(0,1,2))

        print("Dataset mean: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.mean[0], self.mean[1], self.mean[2]))

        print("Dataset std: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.std[0], self.std[1], self.std[2]))

    def standardize(self, img):
        return (img - self.mean) / self.std

    def process_path(self, path):
        img = tf.io.read_file(path)
        img = self.convert_img(img, True)
        return img[:,:,0:1], img[:,:,1:3]

    def convert_img(self, img, standardize):
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [hp.img_size, hp.img_size])
        img = rgb_to_lab(img)        
        return self.standardize(img) if standardize else img

    def get_data(self, path, shuffle):
        # Approach informed by https://www.tensorflow.org/tutorials/load_data/images
        imgs = tf.data.Dataset.list_files([path + "/*.jpg", path + "/*.jpeg", path + "/*.png"])
        cnn_ds = imgs.map(self.process_path, num_parallel_calls=AUTOTUNE)
        cnn_ds = cnn_ds.cache("tf_cache")
        cnn_ds = cnn_ds.shuffle(buffer_size=1000)
        # cnn_ds = cnn_ds.repeat()
        cnn_ds = cnn_ds.batch(hp.batch_size)
        cnn_ds = cnn_ds.prefetch(buffer_size=AUTOTUNE)
        return cnn_ds
