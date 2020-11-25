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
from modules import *


def unet(pretrained_weights=None, input_size=(672, 672, 2), loss="binary_crossentropy", model_name='model_plot.png', spatial_dropout = 0.2, learning_rate = 1e-4, metrics = ['accuracy','TruePositives','TrueNegatives','FalsePositives','FalseNegatives']):
    # Input and conv block 1
    inputs = Input(input_size)

    drop1, pool1 = double_conv_block(inputs, 16, 3, spatial_dropout = spatial_dropout)

    norm2, pool2 = double_conv_block(pool1, 32, 3)

    norm3, pool3 = double_conv_block(pool2, 64, 3)

    norm4, pool4 = double_conv_block(pool3, 128, 3)

    norm5 = double_conv_block(pool4, 256, 3, maxpooling=False)

    norm6 = up_conv_block(norm5, norm4, 128, 3)

    norm7 = up_conv_block(norm6, norm3, 64, 3)

    norm8 = up_conv_block(norm7, norm2, 32, 3)

    norm9 = up_conv_block(norm8, drop1, 16, 3)

    # Output
    conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(norm9)

    model = Model(inputs=inputs, outputs=conv10, name=model_name)

    # compilation
    model.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=metrics)

    model.summary()

    # plot_model(model, to_file=model_name, show_shapes=True)
    
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
        
    model.save(model_name)
    
    return model
