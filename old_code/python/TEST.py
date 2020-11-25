import numpy as np
from model_block import *
from dice_loss import *
from transform import *
import cv2
import matplotlib

def model(files):
    cpt=1
    model = unet(input_size=(320, 320, 1), loss=dice_loss, model_name=f"save_model_scheme/avc_unet3.png")
    model.load_weights('models/model_28/model_28_lowres.h5')
    for inputs in files[2]:
        print(files[0]+'/'+inputs+'.jpg')
        image=cv2.imread(files[0]+'/'+inputs, 1)
        print(image.shape)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = cv2.resize(image, (320, 320))
        image =image /255
        image= np.expand_dims(image, axis=2)
        
        # image=np.concatenate((image, np.flip(image, 1)),axis=2)
        image= np.expand_dims(image, axis=0)

        x=model.predict(image)
        x=np.squeeze(x)*255
        matplotlib.image.imsave(files[0]+'/output_'+str(inputs)+'.png', x)
        cpt+=1