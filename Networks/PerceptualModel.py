from __future__ import absolute_import, division, print_function, unicode_literals
from Networks.NNInterface import NNInterface
from tensorflow.python.keras.applications import vgg16




import tensorflow as tf

IMG_SIZE = 160 # All images will be resized to 160x160

class PerceptualModel(NNInterface):
    def __init__(self, img_size):
        super().__init__()
        self.__model = vgg16.VGG16(input_shape=(img_size[0], img_size[1], 3))

    def call(self, inputs):
        return self.__model(vgg16.preprocess_input(inputs))

    def compute_output_shape(self, input_shape):
        return self.__model.compute_output_shape(input_shape)