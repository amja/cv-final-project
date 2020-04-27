import tensorflow as tf
import hyperparameters as hp
from tensorflow.keras.layers import \
        Conv2D, MaxPool2D, Dropout, Flatten, Dense, ZeroPadding2D, \
        BatchNormalization, Conv2DTranspose, Softmax

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=hp.learning_rate,
            momentum=hp.momentum)

        self.architecture = [
            # Conv1
            ZeroPadding2D(padding=1),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            ZeroPadding2D(padding=1),
            Conv2D(filters=64, kernel_size=3, strides=2, activation='relu'),
            BatchNormalization(),

            # Conv2
            #  question -- does the size change with padding??
            # for the neural network. Padding does add 1 row/ column
            # of 0s to each side.
            ZeroPadding2D(padding=1),
            Conv2D(filters=128, kernel_size=3, activation='relu'),
            ZeroPadding2D(padding=1),
            Conv2D(filters=128, kernel_size=3, strides=2, activation='relu'),
            BatchNormalization(),

            # Conv3
            ZeroPadding2D(padding=1),
            Conv2D(filters=256, kernel_size=3, activation='relu'),
            ZeroPadding2D(padding=1),
            Conv2D(filters=256, kernel_size=3, activation='relu'),
            ZeroPadding2D(padding=1),
            Conv2D(filters=256, kernel_size=3, strides=2, activation='relu'),
            BatchNormalization(),

            # Conv4
            # dilation is spaces between the values in the kernel
            # dilation 1 is the same but I left this in to emphasize differnce
            # 3 x 3 kernel with dilation 2 will have same field of view as
            # 5 x 5 kernel 
            ZeroPadding2D(padding=1),
            Conv2D(filters=512, kernel_size=3, dilation_rate=1, activation='relu'),
            ZeroPadding2D(padding=1),
            Conv2D(filters=512, kernel_size=3, dilation_rate=1, activation='relu'),
            ZeroPadding2D(padding=1),
            Conv2D(filters=512, kernel_size=3, dilation_rate=1, activation='relu'),
            BatchNormalization(),

            # Conv5
            ZeroPadding2D(padding=2),
            Conv2D(filters=512, kernel_size=3, dilation_rate=2, activation='relu'),
            ZeroPadding2D(padding=2),
            Conv2D(filters=512, kernel_size=3, dilation_rate=2, activation='relu'),
            ZeroPadding2D(padding=2),
            Conv2D(filters=512, kernel_size=3, dilation_rate=2, activation='relu'),
            BatchNormalization(),

            # Conv6
            ZeroPadding2D(padding=2),
            Conv2D(filters=512, kernel_size=3, dilation_rate=2, activation='relu'),
            ZeroPadding2D(padding=2),
            Conv2D(filters=512, kernel_size=3, dilation_rate=2, activation='relu'),
            ZeroPadding2D(padding=2),
            Conv2D(filters=512, kernel_size=3, dilation_rate=2, activation='relu'),
            BatchNormalization(),

            # Conv7
            ZeroPadding2D(padding=1),
            Conv2D(filters=512, kernel_size=3, dilation_rate=1, activation='relu'),
            ZeroPadding2D(padding=1),
            Conv2D(filters=512, kernel_size=3, dilation_rate=1, activation='relu'),
            ZeroPadding2D(padding=1),
            Conv2D(filters=512, kernel_size=3, dilation_rate=1, activation='relu'),
            BatchNormalization(),

            # Conv8
            # should have dimension 256 x 64 x 64
            # NEEDS UPSAMPLING -- using transpose, aka Deconvolution
            ZeroPadding2D(padding=1),
            Conv2DTranspose(filters=256, kernel_size=4, dilation_rate=1, strides=2, activation='relu'),
            ZeroPadding2D(padding=1),
            Conv2D(filters=256, kernel_size=3, dilation_rate=1, activation='relu'),
            ZeroPadding2D(padding=1),
            Conv2D(filters=256, kernel_size=3, dilation_rate=1, activation='relu'),

            # Unary prediction
            # this is the (a,b) distribution
            Conv2D(filters=313, kernel_size=1, dilation_rate=1, activation='relu'),

            # REBALANCE LAYER

            # SOFTMAX
            # alternatively activation=tf.nn.softmax
            # if we can figure out from rebalance layer
            Softmax()
        ]
    
    def call(self, img):
        for layer in self.architecture:
            img = layer(img)
        return img

    @staticmethod
    def loss_fn(truth, prediction):
        
        
        
        return 0
        # return tf.keras.losses.sparse_categorical_crossentropy(
        #     labels, predictions, from_logits=False)
