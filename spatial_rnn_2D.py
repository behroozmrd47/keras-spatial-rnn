import tensorflow as tf
import numpy as np
from keras.layers import *
import tensorflow.keras.backend as K
from collections import OrderedDict


class SpatialRNN2D(tf.keras.layers.Layer):
    def __init__(self, rnn_radius, direction='all'):
        """
        Class for Spatial RNN layer capable of learning spatial connections between pixels of an 2D image in an RNN
        fashion along all four directions of up, downs,left and right. The RNN unit is plain RNN with ReLu activation
        function instead of tanh as suggested by Li et. al. (2019). Implemented in tensorflow 2.0 with Keras API.
        If `direction` is 'all', the spatial connections will be analysed in all principal directions of "left",
        "right", "up", "down". If `direction` is only one of the principal direction ("left", "right", "up", "down"),
        spatial connections will only be analysed in the defined direction. It's worth mentioning that The current
        implementation only works for 2D images and training batch size of 1. The input 2D image is recommended to
        be square as sufficient testing with non-square input images has not been done. When using this layer as the
        first layer, preceded it with an Keras "Input" layer. Should be used with `data_format="channels_last"`.

        Examples:
        The inputs are 5x5 RGB images with `channels_last` and the batch of 1
        input_shape = (1, 5, 5, 3)
        x_in = tf.keras.layers.Input((5, 5, 3))
        spatial_rnn = SpatialRNN2D(rnn_radius=4, direction='left')
        y_out = spatial_rnn(x_in)  # output shape of (1, 5, 5, 3)

        Examples:
        The inputs are 5x5 RGB images with `channels_last` and the batch of 1
        input_shape = (1, 5, 5, 3)
        x_in = tf.keras.layers.Input((5, 5, 3))
        spatial_rnn = SpatialRNN2D(rnn_radius=4, direction='all')
        y_out = spatial_rnn(x_in)  # output shape of (1, 5, 5, 12)


          Arguments:
            rnn_radius: Integer, the length of pixels sequence to be analysed by RNN unit
            direction='all': either "all" (default) or one of "left", "right", "up", "down"
          Input shape:
            4D tensor with shape: `batch_shape=1 + (rows, cols, channels)` for `data_format='channels_last'`.
          Output shape:
            4D tensor with shape: `batch_shape=1 + (new_rows, new_cols, filters)` for `data_format='channels_last'`.
          Raises:
            Nothing at the moment. But shall be added.
          """
        super().__init__()
        self.padding = "SAME"
        self.r = rnn_radius
        self.direction = direction.lower()

        self.kernel_switch_dic = OrderedDict({'left': [[1, 0, 0]], 'right': [[0, 0, 1]], 'up': [[1], [0], [0]],
                                              'down': [[0], [0], [1]]})
        if self.direction == 'all':
            self.kernel_switches = list(self.kernel_switch_dic.values())
        else:
            self.kernel_switches = list([self.kernel_switch_dic[direction]])

    def build(self, input_shape):
        """
          Build the class based on the input shape and the direction parameter. The required kernels are built as well.

          Arguments:
            input_shape: 4D tensor with shape: `1 + (rows, cols, channels)`.
          Raises:
            Nothing at the moment. But shall be added.
          """
        self.num_channel = int(input_shape[-1])
        self.hex_filter_list = []
        self.kernel_list = []
        for kernel_switch in self.kernel_switches:
            kernel_switch = np.array(kernel_switch)
            self.hex_filter = self.get_kernel_switch(kernel_switch)
            self.kernel_list.append(self.add_weight(
                shape=[kernel_switch.shape[0], kernel_switch.shape[1], self.num_channel, self.num_channel],
                initializer="random_normal", trainable=True) * self.hex_filter)
            super().build(input_shape)

    def call(self, input_tensor, **kwargs):
        """
          Calls the tensor for forward pass operation.

          Arguments:
            input_tensor: Since at the moment the layer only works with batch of 1 the input_tensor should have shape of
            a 4D tensor with shape: `1 + (rows, cols, channels)`.
          Returns:
            4D tensor representative of the forward pass of the Spatial RNN layer with
            shape: `batch_shape=1 + (new_rows, new_cols, filters)`.
          Raises:
            Nothing at the moment. But shall be added.
          """
        input_tensor = K.cast(tf.identity(input_tensor), tf.float32)
        result_tensors_list = []
        for ker in self.kernel_list:
            res_sum = tf.identity(input_tensor)
            tensor = tf.identity(input_tensor)
            for i in range(self.r):
                conv = tf.nn.conv2d(tensor, ker, strides=[1], padding=self.padding)
                tensor = tf.nn.relu(conv)
                res_sum += tensor
            result_tensors_list.append(res_sum)
        if len(result_tensors_list) == 1:
            return result_tensors_list[0]
        else:
            return concatenate(result_tensors_list, axis=3)

    def compute_output_shape(self, input_shape):
        """
          Compute output shape.

          Arguments:
            input_shape: 4D tensor with shape: `1 + (rows, cols, channels)`.
          Returns:
            4D tensor with shape: `batch_shape=1 + (new_rows, new_cols, filters * 4) if "all" direction is selected or
            4D tensor with shape: `batch_shape=1 + (new_rows, new_cols, filters) if single direction is selected.
          """
        if self.direction == 'all':
            return input_shape[0], input_shape[1], input_shape[2], self.num_channel * 4
        else:
            return input_shape[0], input_shape[1], input_shape[2], self.num_channel

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
        return tf.constant(kernel_switch, dtype=tf.float32)
