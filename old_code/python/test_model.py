import skimage.io as io
import skimage.transform as trans
from model_block import *
import cv2
import numpy as np
from dice_loss import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = io.imread('test/images/3.jpg', as_gray=True)
mask = io.imread('test/masks/3.jpg', as_gray=True)
diceloss = DiceLoss()
model = unet(input_size=(672, 672, 1), loss=diceloss, model_name=f"models/model_28/avc_unet4.png", learning_rate = 1e-5)
model.load_weights('models/model_28/model_28_lowres.h5')

in_img = trans.resize(img,(672,672))
in_img = np.expand_dims(in_img, 2)
# in_img = np.expand_dims(in_img, 0)
print(in_img.shape)
#in_img = np.concatenate((in_img, np.flip(in_img, 1)), axis = 2)
in_img = in_img/255

test_image = model.predict(in_img[tf.newaxis,...])

print(test_image.shape)
test_image = trans.resize(np.squeeze(np.squeeze(test_image*255,axis=3), axis=0),(672,672))
test_image = test_image.astype(int)
print('test image:')
print( test_image.shape)

print('mask')
print(mask.shape)
fig, axs = plt.subplots(1,2)
axs[0].imshow(mask, cmap='gray')
axs[1].imshow(test_image, cmap='gray')