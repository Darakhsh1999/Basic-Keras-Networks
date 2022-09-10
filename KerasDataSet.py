from keras.datasets import mnist
import numpy as np


def VAE_data():

    ''' Data used for BasicVAE code
        Outsputs: (x_train, x_test) '''

    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype('float32') / 255
    x_test = x_test.reshape(x_test.shape + (1,))

    return (x_train, x_test)