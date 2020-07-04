from __future__ import absolute_import, division, print_function, unicode_literals
from Networks.NNInterface import NNInterface
from tensorflow.python.keras.applications import vgg16




import tensorflow as tf


class PerceptualModel(NNInterface):
    def __init__(self):
        super().__init__()
        self.__model = vgg16.VGG16(weights='imagenet')

    def call(self, inputs, training=True):
        return self.__model(vgg16.preprocess_input(inputs), training=training)

    def compute_output_shape(self, input_shape):
        return self.__model.compute_output_shape(input_shape)