
# CA JE SUIS SUR QUE CA MARCHE MAIS C'EST BOURBIER
# import keras
import tensorflow as tf
import numpy as np
from keras import backend as K


def dice_coef(y_true, y_pred):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    smooth = 1

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    result = (2. * intersection + smooth) / \
        (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)
    return K.mean(result)


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


# CETTE PARTIE EST CODEE A LA MAIN MAIS SANS TEST DONC J'ESPERE J'AI PAS FAIS D'ERREUR
class DiceLoss:

    # def __init__(self, smooth=1, weights=[1, 2], norm_weights=True):
    def __init__(self, smooth=1):
        self.smooth = smooth
        self.__name__ = 'nikeras'

    def dice_coef(self, y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        result = (2. * intersection + self.smooth) / \
            (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + self.smooth)
        return K.mean(result)

    def dice_loss(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    def __call__(self, y_true, y_pred):
        return self.dice_loss(y_true, y_pred)
