from keras import backend as K
import numpy as np
from tensorflow import size
from tensorflow.dtypes import cast, float32

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def tversky(y_true, y_pred, smooth=0.1, alpha=0.01):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=0.4):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)


# def wiou_coef(y_true, y_pred, cw = 0.001, smooth=0.01):
#     """
#     IoU = (|X &amp; Y|)/ (|X or Y|)
#     """
    
#     #npix = cast(size(y_true), dtype = float32)
    
    
#     y_true_pos = K.flatten(y_true)
#     y_pred_pos = K.flatten(y_pred)
#     pospix =  K.sum(y_true_pos)
#     print(pospix)
#     bgw = 1 - cw
#     true_pos = K.sum(y_true_pos * y_pred_pos)
#     true_neg = K.sum(1-true_pos)
#     # cw = pospix/npix
#     # bgw = (npix-pospix)/npix
#     false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
#     false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    
#     c = true_pos/(true_pos + false_neg + false_pos)
#     bg = true_neg/(true_neg + false_neg + false_pos)
    
#     WOuI = cw * c + bgw * bg
    
#     return WOuI

# def iou_coef_loss(y_true, y_pred):
#     return - wiou_coef(y_true, y_pred)

def iou_coef(y_true, y_pred, smooth=1):
    """
    IoU = (|X &amp; Y|)/ (|X or Y|)
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    return (intersection + smooth) / ( union + smooth)

def iou_coef_loss(y_true, y_pred):
    return -iou_coef(y_true, y_pred)


def mean_iou(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32') # .5 is the threshold
    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) - inter
    return K.mean((inter + K.epsilon()) / (union + K.epsilon()))