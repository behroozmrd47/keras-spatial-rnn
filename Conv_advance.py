import tensorflow as tf
import numpy as np
from matplotlib import pyplot
from keras.layers import *
import tensorflow.keras.backend as K


class Conv2DSpatial(tf.keras.layers.Layer):
    """

    """
    def __init__(self, rnn_radius, direction='all'):
        """

        :param rnn_radius:
        :param direction:
        """
        super().__init__()
        self.padding = "SAME"
        self.r = rnn_radius
        self.direction = direction.lower()

        self.kernel_switch_dic = {'left': [[1, 0, 0]], 'right': [[0, 0, 1]], 'up': [[1], [0], [0]],
                                  'down': [[0], [0], [1]]}
        if self.direction == 'all':
            self.kernel_switches = list(self.kernel_switch_dic.values())
        else:
            self.kernel_switches = list([self.kernel_switch_dic[direction]])

    def build(self, input_shape):
        """

        :param input_shape:
        :return:
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
            # self.kernel_list.append(tf.Variable(tf.random.normal(
            #     [kernel_switch.shape[0], kernel_switch.shape[1], self.num_channel, self.num_outputs],
            #     stddev=1. / 7.), trainable=True) * self.hex_filter)
            # self.kernel_list[-1] = tf.ones_like(self.kernel_list[-1]) * self.hex_filter
            super().build(input_shape)

    def call(self, input_tensor):
        """

        :param input_tensor:
        :return:
        """
        kernel = self.kernel_list[0]
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
        # return input_tensor

    def compute_output_shape(self, input_shape):
        """

        :param input_shape:
        :return:
        """
        if self.direction == 'all':
            return input_shape[0], input_shape[1], input_shape[2], self.num_channel * 4
        else:
            return input_shape[0], input_shape[1], input_shape[2], self.num_channel

    def get_kernel_switch(self, kernel_switch):
        """

        :param kernel_switch:
        :return:
        """
        # kernel_switch = np.array(self.kernel_switch_dic[direction])
        kernel_switch = np.repeat(kernel_switch[:, :, np.newaxis], int(self.num_channel), axis=-1)
        kernel_switch = np.repeat(kernel_switch[:, :, :, np.newaxis], int(self.num_channel), axis=-1)
        return tf.constant(kernel_switch, dtype=tf.float32)


if __name__ == '__main__':
    image = np.array(range(0, 25)).reshape([1, 5, 5, 1])
    image = np.concatenate((image, image + 25), axis=-1)

    # clas = Conv2DFilterPool(num_outputs=2, rnn_radius=4, direction='all', padding="SAME")
    # clas.build(input_shape=[1, 5, 5, 2])
    # a = clas.call(image)
    # print(a[0, :, :, 0])

    tf.keras.backend.clear_session()
    x_in = tf.keras.layers.Input((5, 5, image.shape[-1]))
    # reshaped = tf.keras.layers.Reshape((28, 28, 1))  # 28x28
    conv1 = Conv2DSpatial(rnn_radius=4, direction='all')  # 14x14
    # conv2 = Conv2DFilterPool(num_outputs=28, padding="valid")  # 6x6
    # conv3 = Conv2DFilterPool(num_outputs=32, padding="same")  # 3x3
    # flatten = tf.keras.layers.Flatten()
    # hidden = tf.keras.layers.Dense(200, activation="tanh")
    # dense = tf.keras.layers.Dense(10, activation="sigmoid")
    #
    # # String them together
    # y_out = dense(hidden(flatten(conv3(conv2(conv1(reshaped(x_in)))))))
    y_out = conv1(x_in)
    model = tf.keras.Model(inputs=x_in, outputs=y_out)

    # clas = Conv2DFilterPool(num_outputs=1, padding='VALID')
    # clas.build(input_shape=[1, 5, 5, 1])
    # clas.call(image)
    # tf.keras.utils.plot_model(model, show_layer_names=False, show_shapes=True, dpi=60)
    model.summary()
    a = model.predict(image)
    print(a)
    print(a.shape)
