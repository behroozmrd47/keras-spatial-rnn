from keras.models import Model
from keras.layers import *
from keras.optimizers import *
import src.Constants as Cns
from src.model_dev.models.Conv_advance import Conv2DSpatial


class Unet2D_res_cr:
    def __init__(self, img_rows=Cns.IMG_ROWS, img_cols=Cns.IMG_COLS, img_depth=1, pretrained_weights=False, **kwargs):
        optimizer = kwargs.get('optimizer', Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199))
        loss = kwargs.get('loss', ['binary_crossentropy'])
        metrics = kwargs.get('metrics', ['accuracy'])

        inputs = Input((img_rows, img_cols, img_depth))  # (256,256,1)
        conv11 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)  # (256,256,32)
        conc11 = concatenate([inputs, conv11], axis=3)  # (256,256,33)
        conv12 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conc11)  # (256,256,32)
        conc12 = concatenate([inputs, conv12], axis=3)  # (256,256,33)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conc12)  # (128,128,33)

        conv21 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)  # (128,128,64)
        conc21 = concatenate([pool1, conv21], axis=3)  # (128,128,97)
        conv22 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conc21)  # (128,128,64)
        conc22 = concatenate([pool1, conv22], axis=3)  # (128,128,97)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conc22)  # (64,64,97)

        conv31 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)  # (64,64,128)
        conc31 = concatenate([pool2, conv31], axis=3)  # (64,64,225)
        conv32 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conc31)  # (64,64,128)
        conc32 = concatenate([pool2, conv32], axis=3)  # (64,64,225)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conc32)  # (32,32,225)

        conv41 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)  # (32,32,256)
        conc41 = concatenate([pool3, conv41], axis=3)  # (32,32,481)
        conv42 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conc41)  # (32,32,256)
        conc42 = concatenate([pool3, conv42], axis=3)  # (32,32,481)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conc42)  # (16,16,481)

        conv51 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)  # (16,16,512)
        conc51 = concatenate([pool4, conv51], axis=3)  # (16,16,993)
        conv52 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conc51)  # (16,16,512)
        conc52 = concatenate([pool4, conv52], axis=3)  # (16,16,993)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conc52), conc42], axis=3)  # (32,32,737)  # (32,32,256)
        conv61 = Conv2D(256, 3, activation='relu', padding='same')(up6)  # (32,32,256)
        conc61 = concatenate([up6, conv61], axis=3)  # (32,32,993)
        conv62 = Conv2D(256, 3, activation='relu', padding='same')(conc61)  # (32,32,256)
        conc62 = concatenate([up6, conv62], axis=3)  # (32,32,993)

        rnn1 = Conv2DSpatial(num_outputs=128, rnn_radius=16, direction='all', padding="SAME")(conv32)  # (64, 64, 128 * 4)
        rnn1_conv32 = Conv2D(128, 1, activation='relu', padding='same', kernel_initializer='he_normal')(rnn1)  # (64,64,128)
        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conc62), rnn1_conv32], axis=3)  # (64,64,256)  # (64,64,128)
        conv71 = Conv2D(128, 3, activation='relu', padding='same')(up7)  # (64,64,128)
        conc71 = concatenate([up7, conv71], axis=3)  # (64,64,384)
        conv72 = Conv2D(128, 3, activation='relu', padding='same')(conc71)  # (64,64,128)
        conc72 = concatenate([up7, conv72], axis=3)  # (64,64,384)

        # up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conc72), conv22], axis=3)
        rnn2 = Conv2DSpatial(num_outputs=64, rnn_radius=32, direction='all', padding="SAME")(conv22)  # (128, 128, 64 * 4)
        rnn2_conv22 = Conv2D(64, 1, activation='relu', padding='same', kernel_initializer='he_normal')(rnn2)  # (128,128,64)
        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conc72), rnn2_conv22], axis=3)  # (128,128,128)  # (128,128,64)
        conv81 = Conv2D(64, 3, activation='relu', padding='same')(up8)  # (128,128,64)
        conc81 = concatenate([up8, conv81], axis=3)  # (128,128,192)
        conv82 = Conv2D(64, 3, activation='relu', padding='same')(conc81)  # (128,128,64)
        conc82 = concatenate([up8, conv82], axis=3)  # (128,128,192)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conc82), conv12], axis=3)
        conv91 = Conv2D(32, 3, activation='relu', padding='same')(up9)
        conc91 = concatenate([up9, conv91], axis=3)
        conv92 = Conv2D(32, 3, activation='relu', padding='same')(conc91)
        conc92 = concatenate([up9, conv92], axis=3)

        conv10 = Conv2D(1, 1, activation='sigmoid')(conc92)

        self.model = Model(inputs, conv10)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        if pretrained_weights:
            self.model.load_weights(pretrained_weights)

    def print_summary(self):
        print(self.model.summary())

    def load_weights(self, path_to_h5_weights):
        # weight_dir = '../weights'
        self.model.load_weights(path_to_h5_weights)
        return self

    def get_model(self):
        return self.model

    def save_model(self, file_path):
        self.model.save(file_path)

    def save_weights(self, file_path):
        self.model.save_weights(file_path, save_format='h5')
