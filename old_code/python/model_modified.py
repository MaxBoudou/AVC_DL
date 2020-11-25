import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def unet(pretrained_weights=None, input_size=(672, 672, 2), loss="binary_entropy", model_name='model_plot.png'):
    # Input and conv block 1
    inputs = Input(input_size)
    conv1_1 = Conv2D(64, 3, activation='relu', padding='same',
                     kernel_initializer='he_normal')(inputs)
    conv1_2 = Conv2D(64, 3, activation='relu', padding='same',
                     kernel_initializer='he_normal')(conv1_1)
    norm1 = BatchNormalization()(conv1_2)
    drop1 = SpatialDropout2D(0.2)(norm1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)

    # Conv block 2
    conv2_1 = Conv2D(128, 3, activation='relu', padding='same',
                     kernel_initializer='he_normal')(pool1)
    conv2_2 = Conv2D(128, 3, activation='relu', padding='same',
                     kernel_initializer='he_normal')(conv2_1)
    #drop2 = SpatialDropout2D(0.2)(conv2_2)
    norm2 = BatchNormalization()(conv2_2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(norm2)

    # Conv block 3
    conv3_1 = Conv2D(256, 3, activation='relu', padding='same',
                     kernel_initializer='he_normal')(pool2)
    conv3_2 = Conv2D(256, 3, activation='relu', padding='same',
                     kernel_initializer='he_normal')(conv3_1)
    #drop3 = SpatialDropout2D(0.2)(conv3_2)
    norm3 = BatchNormalization()(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(norm3)

    # Conv block 4
    conv4_1 = Conv2D(512, 3, activation='relu', padding='same',
                     kernel_initializer='he_normal')(pool3)
    conv4_2 = Conv2D(512, 3, activation='relu', padding='same',
                     kernel_initializer='he_normal')(conv4_1)
    # drop4 = SpatialDropout2D(0.2)(conv4_2)
    norm4 = BatchNormalization()(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2))(norm4)

    # Conv bLock 5
    conv5_1 = Conv2D(1024, 3, activation='relu', padding='same',
                     kernel_initializer='he_normal')(pool4)
    conv5_2 = Conv2D(1024, 3, activation='relu', padding='same',
                     kernel_initializer='he_normal')(conv5_1)
    norm5 = BatchNormalization()(conv5_2)

    # Up block 1
    up1_1 = Conv2DTranspose(512, 4, strides=(
        2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(norm5)
    concat1 = concatenate([norm4, up1_1], axis=3)
    up1_2 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(concat1)
    norm6 = BatchNormalization()(up1_2)
    # drop6 = SpatialDropout2D(0.2)(up1_2)

    # Up block 2
    up2_1 = Conv2DTranspose(256, 4, strides=(
        2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(up1_2)
    concat2 = concatenate([norm3, up2_1], axis=3)
    up2_2 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(concat2)
    norm7 = BatchNormalization()(up2_2)
    # drop7 = SpatialDropout2D(0.2)(up2_2)

    # Up block 3
    up3_1 = Conv2DTranspose(128, 4, strides=(
        2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(norm7)
    concat3 = concatenate([norm2, up3_1], axis=3)
    up3_2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(concat3)
    norm8 = BatchNormalization()(up3_2)
    # drop8 = SpatialDropout2D(0.2)(up3_2)

    # Up block 4
    up4_1 = Conv2DTranspose(64, 4, strides=(
        2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(norm8)
    concat4 = concatenate([drop1, up4_1], axis=3)
    up4_2 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(concat4)
    norm9 = BatchNormalization()(up4_2)
    # drop9 = SpatialDropout2D(0.2)(up4_2)

    # Output
    conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(norm9)

    model = Model(inputs=inputs, outputs=conv10, name='model1')

    model.compile(optimizer=Adam(lr=1e-4), loss=loss, metrics=['accuracy'])

    model.summary()

    plot_model(model, to_file=model_name, show_shapes=True)

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
