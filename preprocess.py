import os
import random
import numpy as np
import tensorflow as tf
import hyperparameters as hp
from lab_funcs import rgb_to_lab, lab_to_rgb
from sklearn.cluster import KMeans, MiniBatchKMeans
import pickle
from scipy.spatial.distance import cdist
import numpy as np
from matplotlib import pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE


class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path):
        assert(hp.img_size % 4 == 0)
        self.q_init = False

        self.train_path = os.path.join(data_path, "train")
        self.test_path = os.path.join(data_path, "test")
        self.file_list = self.get_file_list()

        # Mean and std for standardization
        self.mean = np.zeros((3,))
        self.std = np.ones((3,))
        self.calc_mean_and_std()
        self.cc = self.quantize_colors()

        # Setup data generators
        self.train_data = self.get_data(self.train_path)
        self.test_data = self.get_data(self.test_path)
        self.file_list = None

    '''Network output back into RGB image '''

    def test_pipeline(self, path):
        self.init_q_conversion()
        # Convert to L and quantised colour values
        l, q = self.process_path(path)

        # Convert from quantised back to ab
        ab = tf.reshape(self.get_img_ab_from_q_color(q), [hp.img_size // 4, hp.img_size // 4, 2])
        # Upscale ab to img_size
        ab = tf.image.resize(ab, [hp.img_size, hp.img_size], method="bicubic")
        # Reverse standardisation
        lab = tf.concat([l, ab], axis=2) * self.std + self.mean

        rgb = lab_to_rgb(lab)
        plt.imshow(rgb)
        plt.show()
        return rgb

    '''Takes output in q colors to a rank-4 tensor of ab colors
    q - 3136x313 output of our network
    l - 224x224x1 lightness
    '''

    def model_output_to_tensorboard(self, l, q):
        ab = tf.reshape(self.get_img_ab_from_q_color(q), [hp.img_size // 4, hp.img_size // 4, 2])
        # Upscale ab to img_size
        ab = tf.image.resize(ab, [hp.img_size, hp.img_size], method="bicubic")
        # Reverse standardisation
        lab = tf.concat([l, ab], axis=2) * self.std + self.mean
        rgb = lab_to_rgb(lab)
        # reshaped = tf.reshape(rgb, (-1, 224, 224, 3))
        return rgb

    def get_file_list(self):
        file_list = []
        for root, _, files in os.walk(self.train_path):
            for name in files:
                down = name.lower()
                if down.endswith("jpg") or down.endswith("jpeg") or down.endswith("png"):
                    file_list.append(os.path.join(root, name))
        random.shuffle(file_list)
        return file_list

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
            # Randomly choose 5000*2 ab values from the input images
            num_samples = 5000
            pixels_per_im = 3
            ab_list = []
            # Import images
            for i, file_path in enumerate(self.file_list[:int(num_samples/2)]):
                img = tf.io.read_file(file_path)
                # img now in LAB
                img = self.convert_img(img)
                # get img to just Ab
                ab_image = img[:, :, 1:]

                for j in range(pixels_per_im):
                    ab_list.append(ab_image[random.randrange(0, hp.img_size)][random.randrange(0, hp.img_size)].numpy())

            rand_abs = np.stack(ab_list)
            cc = self.gen_q_cc(rand_abs)
        return cc

    ''' From available images generate 313 cluster centers of ab colors'''

    def gen_q_cc(self, ab_colors):
        print('Generating q colors through kmeans!')
        kmeans = MiniBatchKMeans(n_clusters=313, init_size=313, max_iter=300).fit(ab_colors)
        pickle.dump(kmeans.cluster_centers_, open("qcolors_cc.pkl", "wb"))
        print('...Done.')
        return kmeans.cluster_centers_

    '''
    Goes from Y (224 x 224 x 2) to Z (56 x 56 x 313)
    This gets Z, i.e. the true Q distribution, for each pixel from Y, an ab color image'''

    def get_img_q_color_from_ab(self, ab_img):
        if not self.q_init:
            print("get_img_q_color_from_ab: Q conversion not initialised")
            exit(1)

        # Downscale to 56x56 and reshape
        abs_reshaped = tf.reshape(ab_img[::4, ::4, :], (-1, 2))
        dists, closest_qs = self.nearest_neighbours(abs_reshaped, self.cc, hp.n_neighbours)

        # weight the dists using Gaussian kernel sigma = 5
        # got these 2 lines from /colorization/resources/caffe_traininglayers.py
        wts = tf.exp(-dists**2/(2*5**2))
        wts = wts/tf.math.reduce_sum(wts, axis=1)[:, tf.newaxis]
        indices = tf.concat([self.q_indices, tf.reshape(closest_qs, [-1, 1])], axis=1)

        # Add weights to appropriate positions
        return tf.tensor_scatter_nd_add(
            tf.zeros((abs_reshaped.shape[0], 313)), indices, tf.reshape(wts, [-1]))

    '''
    A tensorflow implementation of nearest neighbours to be used when computing ground truth
    data for the loss function.
    '''

    def nearest_neighbours(self, points, centers, k):
        cents = tf.cast(centers, tf.float32)
        # adapted from https://gist.github.com/mbsariyildiz/34cdc26afb630e8cae079048eef91865

        p_x = tf.reshape(points[:, 0], (-1, 1))
        p_y = tf.reshape(points[:, 1], (-1, 1))
        c_x = tf.reshape(cents[:, 0], (-1, 1))
        c_y = tf.reshape(cents[:, 1], (-1, 1))

        p_x2 = tf.reshape(tf.square(p_x), (-1, 1))
        p_y2 = tf.reshape(tf.square(p_y), (-1, 1))
        c_x2 = tf.reshape(tf.square(c_x), (1, -1))
        c_y2 = tf.reshape(tf.square(c_y), (1, -1))

        dist_px_cx = p_x2 + c_x2 - 2*tf.matmul(p_x, c_x, False, True)
        dist_py_cy = p_y2 + c_y2 - 2*tf.matmul(p_y, c_y, False, True)

        dist = tf.sqrt(dist_px_cx + dist_py_cy)
        dists, inds = tf.nn.top_k(-dist, hp.n_neighbours)
        return -dists, tf.cast(inds, tf.int32)

    '''
    Goes from Z hat (58x58x313) to Y hat (58x58x2)
    Gets the annealed mean or mode.
    See https://github.com/richzhang/colorization/blob/815b3f7808f8f2d9d683e9ed6c5b0a39bec232fb/colorization/demo/colorization_demo_v2.ipynb
    Then upscale from 58x58x2 to 224 x 224
    After the neural network -> can use NP functions
    '''

    def get_img_ab_from_q_color(self, q_img):
        # Determine degree of annealed mean vs pure mean vs mode
        temp = 0.38

        nom = tf.math.exp(tf.math.log(q_img)/temp)
        denom = tf.reshape(tf.tile(tf.reduce_sum(tf.math.exp(tf.math.log(
            tf.reshape(tf.math.top_k(q_img, 5)[0], [-1, 5])/temp)), 1), [313]), [-1, 313])
        # Annealed function
        f = nom/denom

        mean = tf.reduce_mean(tf.reshape(tf.math.top_k(q_img, 5)[0], (-1, 5)), 1)
        ab_img = self.cc[tf.expand_dims(tf.math.argmin(tf.math.abs(
            f - tf.reshape(tf.tile(mean, [313]), [-1, 313])), axis=1), 1)]

        return ab_img

    def calc_mean_and_std(self):
        # just for testing!
        if os.path.isfile('mean.pkl'):
            # Q colors are the cluster centers
            self.mean = pickle.load(open("mean.pkl", "rb"))
            self.std = pickle.load(open("std.pkl", "rb"))
        else:
            # Allocate space in memory for images
            data_sample = np.zeros(
                (hp.preprocess_sample_size, hp.img_size, hp.img_size, 3))
            # Import images
            for i, file_path in enumerate(self.file_list[:hp.preprocess_sample_size]):
                img = self.process_path(file_path, False, quantise=False)
                data_sample[i] = img

            self.mean = np.mean(data_sample, axis=(0, 1, 2))
            self.std = np.std(data_sample, axis=(0, 1, 2))

            print("Dataset mean: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
                self.mean[0], self.mean[1], self.mean[2]))
            print("Dataset std: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
                self.std[0], self.std[1], self.std[2]))

            # delete later, just for testing!
            pickle.dump(self.mean, open("mean.pkl", "wb"))
            pickle.dump(self.std, open("std.pkl", "wb"))

    '''
    Perform preprocessing for image path. To be used by Dataset to prepare ground truth for the loss function.
    Process: rgb => lab => l,ab => l,q
    '''

    def process_path(self, path, split=True, quantise=True):
        img = tf.io.read_file(path)
        img = self.convert_img(img)
        if split:
            # Gets Q colours downsized if required.
            return (img[:, :, 0:1], self.get_img_q_color_from_ab(img[:, :, 1:3]) if quantise else img[:, :, 1:3])
        else:
            return img

    '''
    Resizes, standardises and converts image from rgb to lab.
    '''

    def convert_img(self, img, standardize=True):
        img = tf.image.decode_image(img, channels=3, expand_animations=False, dtype="float32")
        img = tf.image.resize(img, [hp.img_size, hp.img_size])
        img = rgb_to_lab(img)
        return ((img - self.mean) / self.std) if standardize else img

    '''
    Precompute constants to minimise memory usage when quantising images
    '''

    def init_q_conversion(self):
        if not self.q_init:
            self.q_indices = tf.repeat(
                tf.reshape(tf.range((hp.img_size // 4) ** 2, dtype="int32"), [-1, 1]), hp.n_neighbours, axis=0)
            self.q_init = True

    def get_data(self, path):
        imgs = tf.data.Dataset.list_files([path + "/*/*.JPEG", path + "/*/*.jpg",
                                           path + "/*/*.jpeg", path + "/*/*.png"])
        self.init_q_conversion()
        cnn_ds = imgs.map(self.process_path, num_parallel_calls=AUTOTUNE)
        cnn_ds = cnn_ds.batch(hp.batch_size)
        cnn_ds = cnn_ds.prefetch(buffer_size=AUTOTUNE)
        return cnn_ds
