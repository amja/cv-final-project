import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import int64
import hyperparameters as hp
from lab_funcs import rgb_to_lab
from sklearn.cluster import KMeans, MiniBatchKMeans
import pickle
from scipy.spatial.distance import cdist

# from sklearn.neighbors import NearestNeighbors

AUTOTUNE = tf.data.experimental.AUTOTUNE
class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path):
        self.q_init = False
        self.train_path = os.path.join(data_path, "train")
        self.test_path = os.path.join(data_path, "test")

        # Mean and std for standardization
        self.mean = np.zeros((3,))
        self.std = np.ones((3,))
        self.calc_mean_and_std()
        self.quantize_colors()
        
        # Setup data generators
        self.train_data = self.get_data(self.train_path, True)
        self.test_data = self.get_data(self.test_path, False)
        example_img = self.process_path('data/test/cocobackground.jpg', split=False)
        self.get_img_q_color_from_ab(self.lab_img_to_ab(example_img))

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
    def quantize_colors(self):
        if os.path.isfile('qcolors_cc.pkl'):
            # Q colors are the cluster centers 
            cc = pickle.load(open("qcolors_cc.pkl", "rb"))
            
        else:
            # Get list of all images in training directory
            file_list = []
            for root, _, files in os.walk(self.train_path):
                for name in files:
                    if name.endswith("jpg") or name.endswith("jpeg") or name.endswith("png"):
                        file_list.append(os.path.join(root, name))

            # Shuffle filepaths
            random.shuffle(file_list)

            # Take sample of file paths 
            file_list = file_list[:500]

            # Randomly choose 1000 ab values from the input images
            rand_abs = np.zeros((1000, 2))

            # Import images
            for i, file_path in enumerate(file_list):
                img = tf.io.read_file(file_path)
                # img now in LAB
                img = self.convert_img(img)
                # get img to just Ab
                ab_image = self.lab_img_to_ab(img)
                # pick two random pixels
                rand_abs[i*2] = ab_image[random.randrange(0, hp.img_size)][random.randrange(0, hp.img_size)].numpy()
                rand_abs[i*2+1] = ab_image[random.randrange(0, hp.img_size)][random.randrange(0, hp.img_size)].numpy()

            cc = self.gen_q_cc(rand_abs)   
        
        # Get Q colors for all training images
        # v: get empirical probability of colors in the quantized ab space
        # training_ims_q_colors = kmeans.predict(ab_colors_arr[:1000])

        # tensor_images = tf.convert_to_tensor(data_sample, dtype=tf.float32)
        # quant_vals = tf.quantization.quantize(tensor_images, min_range=-110, max_range=110,T=tf.qint8)
        
        # Get probability dist for each Q color (v)
        # v = training_ims_q_colors
        return cc

    ''' From available images generate 313 cluster centers of ab colors'''
    def gen_q_cc(self, ab_colors):
        print('Generating q colors through kmeans!')
        kmeans = MiniBatchKMeans(n_clusters=313, init_size=313, max_iter=100).fit(ab_colors)
        pickle.dump(kmeans.cluster_centers_, open("qcolors_cc.pkl", "wb"))
        print('...Done.')
        return kmeans

    '''
    Goes from Y (224 x 224 x 2) to Z (56 x 56 x 313)
    This gets Z, i.e. the true Q distribution, for each pixel from Y, an ab color image'''
    def get_img_q_color_from_ab(self, ab_img):
        if not self.q_init:
            print("get_img_q_color_from_ab: Q conversion not initialised")
            exit(1)

        # this is the index of the closest
        abs_reshaped = tf.reshape(ab_img[::4, ::4, :], (-1,2))
        dists, closest_qs = self.nearest_neighbours(abs_reshaped, self.cc, hp.n_neighbours)

        # weight the dists using Gaussian kernel sigma = 5
        # got these 2 lines from /colorization/resources/caffe_traininglayers.py
        wts = tf.exp(-dists**2/(2*5**2))
        wts = wts/tf.math.reduce_sum(wts,axis=1)[:,tf.newaxis]
        print("SHAPE: {}".format(tf.reshape(closest_qs, [-1, 1]).shape))
        print("indices: {}".format(self.q_indices.shape))
        indices = tf.concat([self.q_indices, tf.reshape(closest_qs, [-1, 1])], axis=1)
    
        # Add weights to appropriate positions
        return tf.tensor_scatter_nd_add(
            tf.zeros((abs_reshaped.shape[0], 313)), indices, tf.reshape(wts, [-1]))

    def nearest_neighbours(self, points, centers, k):
        cents = tf.cast(centers, tf.float32)
        # adapted from https://gist.github.com/mbsariyildiz/34cdc26afb630e8cae079048eef91865
        
        p_x = tf.reshape(points[:,0], (-1,1))
        p_y = tf.reshape(points[:,1], (-1,1))
        c_x = tf.reshape(cents[:,0], (-1,1))
        c_y = tf.reshape(cents[:,1], (-1,1))
        
        p_x2 = tf.reshape(tf.square(p_x), (-1,1))
        p_y2 = tf.reshape(tf.square(p_y), (-1,1))
        c_x2 = tf.reshape(tf.square(c_x), (1,-1))
        c_y2 = tf.reshape(tf.square(c_y), (1,-1))

        dist_px_cx = p_x2 + c_x2 - 2*tf.matmul(p_x, c_x, False, True)
        dist_py_cy = p_y2 + c_y2 - 2*tf.matmul(p_y, c_y, False, True)

        dist = tf.sqrt(dist_px_cx + dist_py_cy)
        dists, inds = tf.nn.top_k(dists, 5)
        return dists, tf.cast(inds, tf.int64)
    '''
    Goes from Z hat (58x58x313) to Y hat (58x58x2)
    Gets the annealed mean or mode.
    See https://github.com/richzhang/colorization/blob/815b3f7808f8f2d9d683e9ed6c5b0a39bec232fb/colorization/demo/colorization_demo_v2.ipynb
    Then upscale from 58x58x2 to 224 x 224
    ''' 
    def get_img_ab_from_q_color(self, q_img):
        # NOT DONE YET
        return q_img
    
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
            img = self.process_path(file_path, False, quantise=False)
            data_sample[i] = img

        self.mean = np.mean(data_sample, axis=(0,1,2))
        self.std = np.std(data_sample, axis=(0,1,2))

        print("Dataset mean: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.mean[0], self.mean[1], self.mean[2]))

        print("Dataset std: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.std[0], self.std[1], self.std[2]))

    def standardize(self, img):
        return (img - self.mean) / self.std

    def process_path(self, path, split=True, quantise=True):
        img = tf.io.read_file(path)
        img = self.convert_img(img, True)
        if split:
            # Gets Q colours downsized if required.
            return (img[:,:,0:1], self.get_img_q_color_from_ab(img[:,:,1:3]) if quantise else img[:,:,1:3])
        else:
            return img

    def convert_img(self, img, standardize=False):
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [hp.img_size, hp.img_size])
        img = rgb_to_lab(img)        
        return self.standardize(img) if standardize else img

    def lab_img_to_ab(self, lab_img):
        return lab_img[:,:,1:]
    
    def init_q_conversion(self):
        if not self.q_init:
            assert(hp.img_size % 4 == 0)
            scaled_size = hp.img_size // 4
            self.q_indices = tf.repeat(
                tf.reshape(tf.range(scaled_size ** 2, dtype=int64), [-1, 1]), hp.n_neighbours, axis=0)
            self.cc = pickle.load(open("qcolors_cc.pkl", "rb"))
            self.q_init = True

    def get_data(self, path, shuffle):
        # Approach informed by https://www.tensorflow.org/tutorials/load_data/images
        imgs = tf.data.Dataset.list_files([path + "/*.jpg", path + "/*.jpeg", path + "/*.png"])
        self.init_q_conversion()
        cnn_ds = imgs.map(self.process_path, num_parallel_calls=AUTOTUNE)
        cnn_ds = cnn_ds.cache("tf_cache")
        if shuffle:
            cnn_ds = cnn_ds.shuffle(buffer_size=1000)
        # cnn_ds = cnn_ds.repeat()
        cnn_ds = cnn_ds.batch(hp.batch_size)
        cnn_ds = cnn_ds.prefetch(buffer_size=AUTOTUNE)
        return cnn_ds
