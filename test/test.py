import tensorflow as tf
import numpy as np
from matplotlib import pyplot
from keras.layers import *
import tensorflow.keras.backend as K
from collections import OrderedDict
from spatial_rnn_2D import SpatialRNN2D

if __name__ == '__main__':
    image = np.array(range(0, 25)).reshape([1, 5, 5, 1])
    image = np.concatenate((image, image + 25), axis=-1)

    clas = SpatialRNN2D(rnn_radius=1, direction='left')
    clas.build(input_shape=[1, 5, 5, 2])
    a = clas.call(image)
    print(a[0, :, :, 0])

    # tf.keras.backend.clear_session()
    # x_in = tf.keras.layers.Input((5, 5, image.shape[-1]))
    # # reshaped = tf.keras.layers.Reshape((28, 28, 1))  # 28x28
    # conv1 = Conv2DSpatial(rnn_radius=4, direction='all')  # 14x14
    # # conv2 = Conv2DFilterPool(num_outputs=28, padding="valid")  # 6x6
    # # conv3 = Conv2DFilterPool(num_outputs=32, padding="same")  # 3x3
    # # flatten = tf.keras.layers.Flatten()
    # # hidden = tf.keras.layers.Dense(200, activation="tanh")
    # # dense = tf.keras.layers.Dense(10, activation="sigmoid")
    # #
    # # # String them together
    # # y_out = dense(hidden(flatten(conv3(conv2(conv1(reshaped(x_in)))))))
    # y_out = conv1(x_in)
    # model = tf.keras.Model(inputs=x_in, outputs=y_out)
    #
    # # clas = Conv2DFilterPool(num_outputs=1, padding='VALID')
    # # clas.build(input_shape=[1, 5, 5, 1])
    # # clas.call(image)
    # # tf.keras.utils.plot_model(model, show_layer_names=False, show_shapes=True, dpi=60)
    # model.summary()
    # a = model.predict(image)
    # print(a)
    # print(a.shape)
