import tensorflow as tf
import numpy as np
# from matplotlib import pyplot
from keras.layers import *
import tensorflow.keras.backend as K
from collections import OrderedDict
from spatial_rnn_2D import SpatialRNN2D

# Setting up input image arrays & dataset
image1 = np.array(range(0, 50, 2)).reshape([1, 5, 5, 1])
image1 = np.concatenate((image1, image1 + 50), axis=-1)
image2 = np.array(range(100, 150, 2)).reshape([1, 5, 5, 1])
image2 = np.concatenate((image2, image2 + 50), axis=-1)
image_dataset = np.concatenate((image1, image2), axis=0)

# Setting up label arrays & dataset
label1_ch1 = np.array([[0, 52, 158, 172, 186],
                       [10, 82, 228, 242, 256],
                       [20, 112, 298, 312, 326],
                       [30, 142, 368, 382, 396],
                       [40, 172, 438, 452, 466]]).reshape((1, 5, 5, 1))
label1_ch2 = np.array([[50, 102, 208, 222, 236],
                       [60, 132, 278, 292, 306],
                       [70, 162, 348, 362, 376],
                       [80, 192, 418, 432, 446],
                       [90, 222, 488, 502, 516]]).reshape((1, 5, 5, 1))
label1 = np.concatenate((label1_ch1, label1_ch2), axis=-1)

label2_ch1 = np.array([[100, 352, 858, 872, 886],
                       [110, 382, 928, 942, 956],
                       [120, 412, 998, 1012, 1026],
                       [130, 442, 1068, 1082, 1096],
                       [140, 472, 1138, 1152, 1166]]).reshape((1, 5, 5, 1))
label2_ch2 = np.array([[150, 402, 908, 922, 936],
                       [160, 432, 978, 992, 1006],
                       [170, 462, 1048, 1062, 1076],
                       [180, 492, 1118, 1132, 1146],
                       [190, 522, 1188, 1202, 1216]]).reshape((1, 5, 5, 1))
label2 = np.concatenate((label2_ch1, label2_ch2), axis=-1)
label_dataset = np.concatenate((label1, label2), axis=0)

tf.keras.backend.clear_session()
x_in = tf.keras.layers.Input((5, 5, image1.shape[-1]))
sp_rnn_layer = SpatialRNN2D(rnn_radius=2, direction='left')
y_out = sp_rnn_layer(x_in)
model = tf.keras.Model(inputs=x_in, outputs=y_out)
model.summary()


def test_forward_pass():
    test_img1_output = model.predict(image1)
    # print(test_output[0, :, :, 0])
    # print(test_output[0, :, :, 1])
    if not np.array_equal(label1, test_img1_output):
        raise BaseException('Forward pass error!!')


def test_train():
    model.compile(optimizer='SGD', loss=tf.losses.binary_crossentropy)
    model.fit(x=image_dataset, y=label_dataset, batch_size=1, epochs=10)

    # if np.array_equal(sample_out, test_output):
    #     raise BaseException('Forward pass error!!')


if __name__ == '__main__':
    test_forward_pass()
    test_train()

    # image = np.array(range(0, 50, 2)).reshape([1, 5, 5, 1])
    # image = np.concatenate((image, image + 50), axis=-1)
    #
    # clas = SpatialRNN2D(rnn_radius=2, direction='left')
    # clas.build(input_shape=[1, 5, 5, 2])
    # a = clas.call(image)
    # print(a[0, :, :, 0])
    # print(a[0, :, :, 1])
    #
    # tf.keras.backend.clear_session()
    # x_in = tf.keras.layers.Input((5, 5, image.shape[-1]))
    # reshaped = tf.keras.layers.Reshape((28, 28, 1))  # 28x28
    # conv1 = Conv2DSpatial(rnn_radius=4, direction='all')  # 14x14
    # conv2 = Conv2DFilterPool(num_outputs=28, padding="valid")  # 6x6
    # conv3 = Conv2DFilterPool(num_outputs=32, padding="same")  # 3x3
    # flatten = tf.keras.layers.Flatten()
    # hidden = tf.keras.layers.Dense(200, activation="tanh")
    # dense = tf.keras.layers.Dense(10, activation="sigmoid")
    #
    # # String them together
    # y_out = dense(hidden(flatten(conv3(conv2(conv1(reshaped(x_in)))))))
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
