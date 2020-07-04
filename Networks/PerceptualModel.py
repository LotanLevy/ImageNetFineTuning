from __future__ import absolute_import, division, print_function, unicode_literals
from Networks.NNInterface import NNInterface
from tensorflow.python.keras.applications import vgg16




import tensorflow as tf


class PerceptualModel(NNInterface):
    def __init__(self):
        super().__init__()
        self.__model = vgg16.VGG16(weights='imagenet')
        self.network.summary()


    def call(self, x, training=True):
        x = vgg16.preprocess_input(x)
        return self.__model(x, training=training)

    def compute_output_shape(self, input_shape):
        return self.__model.compute_output_shape(input_shape)