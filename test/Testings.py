"""
@author: Behrooz Bajestani (behrooz.mrd47@gmail.com)
"""
import numpy as np
import unittest
from Spatial_RNN_2D import SpatialRNN2D
import tensorflow.keras.backend as K
import tensorflow as tf


class TestBoneExtraction(unittest.TestCase):
    """
    Unittest class including tests Spatial_RNN_2D Module.
    """
    # Setting up input image arrays & dataset
    image1_1D = np.array(range(0, 18, 2)).reshape([1, 3, 3, 1])
    image2_1D = np.array(range(100, 118, 2)).reshape([1, 3, 3, 1])
    image_dataset_1D = np.concatenate((image1_1D, image2_1D), axis=0)

    sample_image1_1D_label = np.array([[[[0., 6., 0., 18.],
                                         [2., 6., 2., 24.],
                                         [6., 4., 4., 30.]],
                                        [[6., 24., 6., 18.],
                                         [14., 18., 10., 22.],
                                         [24., 10., 14., 26.]],
                                        [[12., 42., 18., 12.],
                                         [26., 30., 24., 14.],
                                         [42., 16., 30., 16.]]]])

    def test_forward_pass_one_ch(self):
        """
        Test forward pass for one image sample with size of (3*3*1). The layer kernel is initialized as ones so the
        result array would be sum of two indices to left, right, down and up.
        :return: None
        """

        K.clear_session()
        x_in = tf.keras.layers.Input(TestBoneExtraction.image1_1D.shape[1:])
        y_out = SpatialRNN2D(rnn_seq_length=2, kernel_initializer='ones', merge_mode='concat')(x_in)
        model = tf.keras.Model(inputs=x_in, outputs=y_out)
        test_img1_output = model.predict(TestBoneExtraction.image1_1D)
        self.assertTrue(np.array_equal(TestBoneExtraction.sample_image1_1D_label, test_img1_output))

    def test_forward_pass_multi_ch_concat(self):
        """
        Test forward pass for one image sample with size of (3*3*2). The layer kernel is initialized to ones for both
        channels.
        :return: None
        """

        sample_spatial_rnn_label = np.array([[[[0., 20., 6., 66., 0., 20., 18., 78.],
                                               [2., 42., 6., 46., 2., 22., 24., 84.],
                                               [6., 66., 4., 24., 4., 24., 30., 90.]],
                                              [[6., 26., 24., 84., 6., 46., 18., 58.],
                                               [14., 54., 18., 58., 10., 50., 22., 62.],
                                               [24., 84., 10., 30., 14., 54., 26., 66.]],
                                              [[12., 32., 42., 102., 18., 78., 12., 32.],
                                               [26., 66., 30., 70., 24., 84., 14., 34.],
                                               [42., 102., 16., 36., 30., 90., 16., 36.]]]])

        image1_2d = np.concatenate((TestBoneExtraction.image1_1D, TestBoneExtraction.image1_1D + 20), axis=-1)
        K.clear_session()
        x_in = tf.keras.layers.Input(image1_2d.shape[1:])
        y_out = SpatialRNN2D(rnn_seq_length=3, kernel_initializer='ones', merge_mode='concat')(x_in)
        model = tf.keras.Model(inputs=x_in, outputs=y_out)
        test_img1_output_ch2 = model.predict(image1_2d)
        self.assertTrue(np.array_equal(sample_spatial_rnn_label, test_img1_output_ch2))

    def test_forward_pass_multi_ch_conv(self):
        """
        Test forward pass for one image sample with size of (3*3*2). The layer kernel is initialized to ones for both
        channels.
        :return: None
        """
        sample_spatial_rnn_label = np.array([[[[208.],
                                               [228.],
                                               [248.]],
                                              [[268.],
                                               [288.],
                                               [308.]],
                                              [[328.],
                                               [348.],
                                               [368.]]]])

        image1_2d = np.concatenate((TestBoneExtraction.image1_1D, TestBoneExtraction.image1_1D + 20), axis=-1)
        K.clear_session()
        x_in = tf.keras.layers.Input(image1_2d.shape[1:])
        y_out = SpatialRNN2D(rnn_seq_length=3, kernel_initializer='ones', merge_mode='convolution',
                             output_conv_filter=1)(x_in)
        model = tf.keras.Model(inputs=x_in, outputs=y_out)
        test_img1_output_ch2 = model.predict(image1_2d)
        self.assertTrue(np.array_equal(sample_spatial_rnn_label, test_img1_output_ch2))

    def test_rnn_sequence_length_warning(self):
        """
        Test for rnn sequence length bigger or equal to image edge size. rnn sequence lengths of 3 & 10 for image size
        of 3.
        :return: None
        """
        K.clear_session()
        x_in = tf.keras.layers.Input(TestBoneExtraction.image1_1D.shape[1:])
        y_out_1 = SpatialRNN2D(rnn_seq_length=3, kernel_initializer='ones')(x_in)
        y_out_2 = SpatialRNN2D(rnn_seq_length=10, kernel_initializer='ones')(x_in)
        model1 = tf.keras.Model(inputs=x_in, outputs=y_out_1)
        model2 = tf.keras.Model(inputs=x_in, outputs=y_out_2)
        test_img1_output_1 = model1.predict(TestBoneExtraction.image1_1D)
        test_img1_output_2 = model2.predict(TestBoneExtraction.image1_1D)
        self.assertTrue(np.array_equal(TestBoneExtraction.sample_image1_1D_label, test_img1_output_1))
        self.assertTrue(np.array_equal(test_img1_output_1, test_img1_output_2))

    def test_forward_pass_two_sample_dataset(self):
        """
        Test forward pass for dataset of two images with size of (3*3*1). The layer kernel is initialized to ones.
        :return: None
        """
        sample_img_dataset_output_label = np.array([[[[0., 6., 0., 18.],
                                                      [2., 6., 2., 24.],
                                                      [6., 4., 4., 30.]],
                                                     [[6., 24., 6., 18.],
                                                      [14., 18., 10., 22.],
                                                      [24., 10., 14., 26.]],
                                                     [[12., 42., 18., 12.],
                                                      [26., 30., 24., 14.],
                                                      [42., 16., 30., 16.]]],
                                                    [[[100., 306., 100., 318.],
                                                      [202., 206., 102., 324.],
                                                      [306., 104., 104., 330.]],
                                                     [[106., 324., 206., 218.],
                                                      [214., 218., 210., 222.],
                                                      [324., 110., 214., 226.]],
                                                     [[112., 342., 318., 112.],
                                                      [226., 230., 324., 114.],
                                                      [342., 116., 330., 116.]]]])

        K.clear_session()
        x_in = tf.keras.layers.Input(TestBoneExtraction.image_dataset_1D.shape[1:])
        y_out = SpatialRNN2D(rnn_seq_length=2, kernel_initializer='ones')(x_in)
        model = tf.keras.Model(inputs=x_in, outputs=y_out)
        test_img_dataset_output = model.predict(TestBoneExtraction.image_dataset_1D)
        self.assertTrue(np.array_equal(sample_img_dataset_output_label, test_img_dataset_output))


if __name__ == '__main__':
    unittest.main()
