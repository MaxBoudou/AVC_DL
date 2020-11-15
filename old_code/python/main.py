from model_block import *
from data import *
from new_loss import *
from livelossplot import PlotLossesKerasTF
from transform import *
from shuffle_dataset import *
from transform import *
import os
import pickle
import tensorflow as tf

data_gen_args = dict(
                    # width_shift_range=0.05,
                    height_shift_range=0,
                    zoom_range=0,
                    horizontal_flip=False,
                    vertical_flip = False,
                    fill_mode='nearest')

TEST_TRANSFORM = [
    ToType(np.float),
    # Flip(),
    # Normalize(),   
    Brightness((-2, 2)),
    Contrast((-2, 2)),
    GaussianBlur(max_kernel=(5, 5)),
    GaussianNoise(0, 15),
    # Vignetting(ratio_min_dist=0.1,
    #            range_vignette=(0.1, 0.7),
    #            random_sign=True),
    LensDistortion(d_coef=(0.2, 0.2, 0.1, 0.1, 0.1)),
    Perspective(max_ratio_translation=(0.0, 0.0, 0),
                max_rotation=(0, 0, 0),
                max_scale=(0.1, 0.1, 0.1),
                max_shearing=(0, 0, 0)),
    Clip(mini = 0, maxi = 255),
]

# shuffle_dataset(0.3,'train/','images/','masks/','test/')

dim_image = 352
target_size = (dim_image,dim_image)
train_data_generator = trainGenerator(35,'train/','images','masks',data_gen_args, TEST_TRANSFORM,target_size=target_size, image_save_prefix=None, mask_save_prefix=None, save_to_dir = None)
validation_data_generator = validationGenerator('test/','images/','masks/',target_size=target_size)


model_index = 106
model_path = f"models/model_{model_index}"
if not os.path.exists(model_path):
    os.mkdir(model_path)
    

model = unet(input_size=(dim_image, dim_image, 1), loss=FocalTverskyLoss, model_name=f"{model_path}/model_{model_index}_lowres", spatial_dropout = 0.2, learning_rate = 0.1)

model_checkpoint = ModelCheckpoint(
    model_path+f"/model_{model_index}_lowres.h5", monitor='val_loss', verbose=1, save_best_only=True)


live_loss = PlotLossesKerasTF()

hstry = model.fit(
    x = train_data_generator,
    batch_size = None,
    epochs=30,
    verbose=1,
    callbacks=[model_checkpoint,live_loss],
    validation_split=None,
    validation_data=validation_data_generator,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=59,
    validation_steps=745,
    validation_batch_size=None,
    validation_freq=1,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False
)

with open(f"{model_path}/trainHistoryDict_{model_index}", 'wb') as file_pi:
        pickle.dump(hstry.history, file_pi)



model.load_weights(f"{model_path}/model_{model_index}_lowres.h5")
test_data_generator = testGenerator('test/','images/','masks/',target_size=target_size)
model.evaluate(test_data_generator,verbose=1)
test_data_generator = testGenerator('test/','images/','masks/',target_size=target_size)
results = model.predict(test_data_generator,verbose=1)

if not os.path.exists(f"{model_path}/results_{model_index}"):
    os.mkdir(f"{model_path}/results_{model_index}")
saveResult(f"{model_path}/results_{model_index}",results)
