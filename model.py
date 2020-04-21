import tensorflow as tf
import hyperparameters as hp

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=hp.learning_rate,
            momentum=hp.momentum)

        self.architecture = []
    
    def call(self, img):
        for layer in self.architecture:
            img = layer(img)
        return img

    @staticmethod
    def loss_fn(labels, predictions):
        return None
        # return tf.keras.losses.sparse_categorical_crossentropy(
        #     labels, predictions, from_logits=False)
