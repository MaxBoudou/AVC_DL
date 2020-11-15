from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


# Conv block
def double_conv_block(inputs, filters, kernelsize, activation='relu', padding='same', maxpooling=True, spatial_dropout=False, kernel_initializer='he_normal'):

    conv1 = Conv2D(filters, kernelsize, activation=activation,
                   padding='same', kernel_initializer='he_normal')(inputs)
    conv2 = Conv2D(filters, kernelsize, activation=activation,
                   padding='same', kernel_initializer='he_normal')(conv1)
    norm = BatchNormalization()(conv2)

    if spatial_dropout:
        norm = SpatialDropout2D(spatial_dropout)(norm)

    if maxpooling:
        pool = MaxPooling2D(pool_size=(2, 2))(norm)
        return norm, pool
    else:
        return norm


def up_conv_block(prev_layer_inputs, symetric_inputs, filters, conv_kernelsize ,up_kernelsize=4, strides=(2, 2), activation='relu', padding='same', spatial_dropout=False, kernel_initializer='he_normal'):

    up1 = Conv2DTranspose(filters, up_kernelsize, strides=strides, padding=padding,
                          activation=activation, kernel_initializer='he_normal')(prev_layer_inputs)
    concat2 = concatenate([up1, symetric_inputs], axis=3)
    up2 = Conv2D(filters, conv_kernelsize, activation=activation,
                 padding=padding, kernel_initializer=kernel_initializer)(concat2)
    norm = BatchNormalization()(up2)

    if spatial_dropout:
        drop = SpatialDropout2D(spatial_dropout)(norm)
        return drop

    else:
        return norm
