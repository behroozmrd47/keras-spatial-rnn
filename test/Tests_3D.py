"""
Test script for Bone_extraction Module Feb 29 2020
@author: Behrooz Bajestani
"""
import numpy as np
import unittest
from scipy.io import loadmat
from Spatial_RNN_3D import SpatialRNN3D
from keras.optimizers import *
import tensorflow.keras.backend as K
import tensorflow as tf


class TestBoneExtraction(unittest.TestCase):
    """
    Unittest class including tests Spatial_RNN_2D Module.
    """
    # Setting up input image arrays & dataset
    image1_2D = np.array(range(0, 16, 2)).reshape([1, 2, 2, 2], order='F')
    image2_2D = np.array(range(20, 36, 2)).reshape([1, 2, 2, 2], order='F')
    image3_2D = np.array(range(40, 56, 2)).reshape([1, 2, 2, 2], order='F')
    image4_2D = np.array(range(60, 76, 2)).reshape([1, 2, 2, 2], order='F')

    stack_image_1 = np.concatenate((image1_2D, image2_2D), axis=0)
    stack_image_1 = np.expand_dims(stack_image_1, axis=0)
    stack_image_2 = np.concatenate((image3_2D, image4_2D), axis=0)
    stack_image_2 = np.expand_dims(stack_image_2, axis=0)
    image_dataset_1D = np.concatenate((stack_image_1, stack_image_2), axis=0)

    # sample_image1_1D_label = np.array([[[[0., 6., 0., 18.],
    #                                      [2., 6., 2., 24.],
    #                                      [6., 4., 4., 30.]],
    #                                     [[6., 24., 6., 18.],
    #                                      [14., 18., 10., 22.],
    #                                      [24., 10., 14., 26.]],
    #                                     [[12., 42., 18., 12.],
    #                                      [26., 30., 24., 14.],
    #                                      [42., 16., 30., 16.]]]])

    def test_forward_pass_one_sample_one_ch(self):
        """
        Test forward pass for one image sample with size of (3*3*1). The layer kernel is initialized as ones so the
        result array would be sum of two indices to left, right, down and up.
        :return: None
        """

        K.clear_session()
        x_in = tf.keras.layers.Input(TestBoneExtraction.image1_2D.shape[1:])
        y_out = SpatialRNN3D(rnn_seq_length=1, kernel_initializer='ones')(x_in)
        model = tf.keras.Model(inputs=x_in, outputs=y_out)
        test_img1_output = model.predict(TestBoneExtraction.image1_2D)
        # self.assertTrue(np.array_equal(TestBoneExtraction.sample_image1_1D_label, test_img1_output))

    def test_forward_pass_one_sample_multi_ch(self):
        """
        Test forward pass for one image sample with size of (3*3*2). The layer kernel is initialized to ones for both
        channels.
        :return: None
        """

        sample_spatial_rnn_label = np.array([[[[0., 20., 80., 100., 0., 20., 120., 140.],
                                               [22., 42., 30., 50., 2., 22., 134., 154.],
                                               [68., 88., 4., 24., 4., 24., 148., 168.]],
                                              [[6., 26., 122., 142., 26., 46., 50., 70.],
                                               [40., 60., 48., 68., 32., 52., 56., 76.],
                                               [110., 130., 10., 30., 38., 58., 62., 82.]],
                                              [[12., 32., 164., 184., 84., 104., 12., 32.],
                                               [58., 78., 66., 86., 98., 118., 14., 34.],
                                               [152., 172., 16., 36., 112., 132., 16., 36.]]]])

        image1_2d = np.concatenate((TestBoneExtraction.image1_2D, TestBoneExtraction.image1_2D + 20), axis=-1)
        K.clear_session()
        x_in = tf.keras.layers.Input(image1_2d.shape[1:])
        y_out = SpatialRNN2D(rnn_seq_length=3, kernel_initializer='ones')(x_in)
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
        x_in = tf.keras.layers.Input(TestBoneExtraction.image1_2D.shape[1:])
        y_out_1 = SpatialRNN2D(rnn_seq_length=3, kernel_initializer='ones')(x_in)
        y_out_2 = SpatialRNN2D(rnn_seq_length=10, kernel_initializer='ones')(x_in)
        model1 = tf.keras.Model(inputs=x_in, outputs=y_out_1)
        model2 = tf.keras.Model(inputs=x_in, outputs=y_out_2)
        test_img1_output_1 = model1.predict(TestBoneExtraction.image1_2D)
        test_img1_output_2 = model2.predict(TestBoneExtraction.image1_2D)
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
        x_in = tf.keras.layers.Input((3, 3, 1))
        y_out = SpatialRNN2D(rnn_seq_length=2, kernel_initializer='ones')(x_in)
        model = tf.keras.Model(inputs=x_in, outputs=y_out)
        test_img_dataset_output = model.predict(TestBoneExtraction.image_dataset_1D)
        self.assertTrue(np.array_equal(sample_img_dataset_output_label, test_img_dataset_output))


if __name__ == '__main__':
    unittest.main()
