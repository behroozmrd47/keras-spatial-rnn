import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from collections import OrderedDict
from tensorflow.python.keras import initializers
from tensorflow.python.keras import activations
import warnings


class SpatialRNN3D():
    def __init__(self, rnn_seq_length, activation='relu', kernel_initializer='random_uniform', **kwargs):

        """
        Class for Spatial RNN layer capable of learning spatial connections between pixels of an 2D image in an RNN
        fashion along all four directions of up, downs,left and right. Implemented in tensorflow 2.0 with Keras API.
        The RNN unit is plain RNN with ReLu activation function (default) as suggested by Li et. al. (2019).
        The spatial connections will be analysed in all principal directions sweeping to right, left, down, up.
        The layer is designed to have output shape with same image size as the input shape and 4 times
        the number of channels.
        It's worth mentioning that The current implementation only works for 2D images and training batch size of 1.
        The input 2D image is recommended to be square as sufficient testing with non-square input images has not been
        done. When using this layer as the first layer, preceded it with an Keras "Input" layer. Should be used with
        `data_format="channels_last"`. The kernel initializer and activation functions can be set using the ones
        available in tensorflow.python.keras.initializers & tensorflow.python.keras.activations.

        Examples:
        The inputs are 5x5 RGB images with `channels_last` and the batch of 1
        input_shape = (1, 5, 5, 3)
        x_in = tf.keras.layers.Input((5, 5, 3))
        spatial_rnn = SpatialRNN2D(rnn_seq_length=4)
        y_out = spatial_rnn(x_in)  # output shape of (1, 5, 5, 12)

          Arguments:
            rnn_seq_length: Integer, the length of pixels sequence to be analysed by RNN unit
            activation: Activation function to use. If you don't specify anything, relu activation is applied
            (see `keras.activations`).
            kernel_initializer: Initializer for the `kernel` weights matrix (see `keras.initializers`).
          """
        # super().__init__()
        self.padding = "same"
        self.seq_length = rnn_seq_length
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_dic = OrderedDict()
        self.kernel_switch_dic = OrderedDict({'right': np.array([[1, 0, 0]]),
                                              'left': np.array([[0, 0, 1]]),
                                              'down': np.array([[1], [0], [0]]),
                                              'up': np.array([[0], [0], [1]])})

    def build(self, input_shape):
        """
          Build the class based on the input shape and the direction parameter. The required kernels are built as well.

          Arguments:
            input_shape: 4D tensor with shape: `(batch_shape, rows, cols, channels)`.
          Raises:
            Warning if the rnn sequence length is equal to or greater than input image edge size.
          """
        if self.seq_length >= input_shape[1] or self.seq_length >= input_shape[2]:
            warnings.warn("The rnn sequence length parameter is equal or bigger than image edge size. This will not "
                          "have any effect on the results but will increase computation cost. You can change the "
                          "rnn sequence length parameter to as big as (edge size - 1).")

        self.num_channel = int(input_shape[-1])
        for direction, kernel_switch in self.kernel_switch_dic.items():
            self.kernel_switch_dic[direction] = self.get_kernel_switch(kernel_switch)

        for direction, kernel_switch in self.kernel_switch_dic.items():
            self.kernel_dic[direction] = tf.Variable(tf.ones(kernel_switch.shape), trainable=True)
        # self.add_weight(
        #     shape=kernel_switch.shape, initializer=self.kernel_initializer, trainable=True)

        # super().build(input_shape)


    # @tf.function
    def call(self, input_tensor, **kwargs):
        """
          Calls the tensor for forward pass operation.

          Arguments:
            input_tensor: The input dataset of 2D images with shape of `(batch_shape, rows, cols, channels)`.
          Returns:
            4D tensor representative of the forward pass of the Spatial RNN layer with
            shape: `(batch_shape, input_image_rows, input_image_cols, input_image_channels * 4)`.
          """
        input_tensor = K.cast(tf.identity(input_tensor), tf.float32)
        result_tensors_list = []
        for direction, kernel in self.kernel_dic.items():
            res_sum = tf.identity(input_tensor)
            tensor = tf.identity(input_tensor)
            for i in range(self.seq_length):
                conv = K.conv2d(x=tensor, kernel=kernel * self.kernel_switch_dic[direction], strides=[1],
                                padding='same')
                tensor = self.activation(conv)
                res_sum += tensor
            result_tensors_list.append(res_sum)
        return K.concatenate(result_tensors_list, axis=-1)

    def compute_output_shape(self, input_shape):
        """
          Compute output shape.

          Arguments:
            input_shape: 4D tensor with shape: `1 + (rows, cols, channels)`.
          Returns:
            4D tensor with shape: `(batch_shape, input_image_rows, input_image_cols, input_image_channels * 4)`
          """
        return input_shape[0], input_shape[1], input_shape[2], self.num_channel * 4

    def get_kernel_switch(self, kernel_switch):
        """
          Compute the ker nel switch.

          Arguments:
            kernel_switch: The kernel switch in format of numpy array consisting of zeros and ones.
          Returns:
            The tensor format of kernel switch consisting of zeros and ones.
          """
        kernel_switch = np.repeat(kernel_switch[:, :, np.newaxis], int(self.num_channel), axis=-1)
        kernel_switch = np.repeat(kernel_switch[:, :, :, np.newaxis], int(self.num_channel), axis=-1)
        return K.constant(kernel_switch, dtype=tf.float32)


if __name__ == '__main__':
    image1_2D = np.array(range(0, 16, 2)).reshape([1, 2, 2, 2], order='F')
    image2_2D = np.array(range(20, 36, 2)).reshape([1, 2, 2, 2], order='F')
    image3_2D = np.array(range(40, 56, 2)).reshape([1, 2, 2, 2], order='F')
    image4_2D = np.array(range(60, 76, 2)).reshape([1, 2, 2, 2], order='F')
    image_dataset_2D = np.concatenate((image1_2D, image2_2D), axis=0)

    # stack_image_1 = np.concatenate((image1_2D, image2_2D), axis=0)
    # stack_image_1 = np.expand_dims(stack_image_1, axis=0)
    # stack_image_2 = np.concatenate((image3_2D, image4_2D), axis=0)
    # stack_image_2 = np.expand_dims(stack_image_2, axis=0)
    # image_dataset_3D = np.concatenate((stack_image_1, stack_image_2), axis=0)
    #
    # image1_1D = np.array(range(0, 18, 2)).reshape([1, 3, 3, 1])
    # image2_1D = np.array(range(100, 118, 2)).reshape([1, 3, 3, 1])



    rnn_layer = SpatialRNN3D(rnn_seq_length=1)
    rnn_layer.build(input_shape=image_dataset_2D.shape)
    rnn_layer.call(image_dataset_2D)
