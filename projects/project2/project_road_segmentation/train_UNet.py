# Train models and generate submission csv from scratch
# Models: UNet, UNet with dilated convolution

import os
import argparse
import tensorflow as tf

from metric_loss import dice_loss
from dataset_UNet import data_aug, set_generator
from model.UNet import unet
from model.dilated_UNet import dilated_unet
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# Set random seed
tf.random.set_seed(seed=123)

# Path options
parser = argparse.ArgumentParser(description='U-Net Model Training')
parser.add_argument('--train_path',
                    metavar='data/training',
                    type=str,
                    default='data/training')
parser.add_argument('--val_path',
                    metavar='data/validation',
                    type=str,
                    default='data/validation')
parser.add_argument('--test_path',
                    metavar='data/test_set_images',
                    type=str,
                    default='data/test_set_images'
                    )
parser.add_argument('--weight_path',
                    metavar='weights/',
                    type=str,
                    default='weights/'
                    )
args = parser.parse_args()

# Training parameters
EPOCH = 50
TRAIN_STEP = 1000
VAL_STEP = 80
SUB_SIZE = 50

# File Paths
path_tr = args.train_path
path_te = args.val_path
path_test = args.test_path
path_model = args.weight_path

path_pred = 'pred_imgs'
path_sub = 'submission'
path_segnet = 'data_segnet'


# Parameters for ImageDataGenerator
generator_args = dict(rotation_range=45,
                      width_shift_range=0.1,
                      height_shift_range=0.1,
                      horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='reflect')


if __name__ == '__main__':
    if not os.path.exists(path_te):
        print("Creating training and validation images...")
        data_aug(path_tr, path_te, test_size=0.2)

    else:
        print("Training and validation images found...")

    print("Create training set and validation set...")

    # Build generator for training and validation set
    train_set, test_set = set_generator(batch_size=2, aug_dict=generator_args, path_tr=path_tr, path_te=path_te,
                                        image_folder='images', mask_folder='groundtruth', dir_tr=None, dir_te=None,
                                        target_size=(320, 320), seed=123)


    # UNet32 training
    print("Training UNet with 32 initial filters...")
    model_unet32 = unet(n_filter=32, activation='elu', loss=dice_loss, d_rate=0.2)
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4),
                 ModelCheckpoint(os.path.join(path_model, 'weights_u32.h5'), monitor='val_loss', save_best_only=True,
                                 verbose=1)]
    history_unet32 = model_unet32.fit_generator(generator=train_set, steps_per_epoch=TRAIN_STEP,
                                                validation_data=test_set, validation_steps=VAL_STEP,
                                                epochs=EPOCH, callbacks=callbacks)


    # UNet64 training
    print("Training UNet with 64 initial filters...")
    # Build model
    model_unet64 = unet(n_filter=64, activation='elu', loss=dice_loss, d_rate=0.2)
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4),
                 ModelCheckpoint(os.path.join(path_model, 'weights_u64.h5'), monitor='val_loss', save_best_only=True,
                                 verbose=1)]

    history_unet64 = model_unet64.fit_generator(generator=train_set, steps_per_epoch=TRAIN_STEP,
                                                validation_data=test_set, validation_steps=VAL_STEP,
                                                epochs=EPOCH, callbacks=callbacks)


    # Dilated UNet training
    print("Training the dilated UNet with 64 initial filters...")
    model_dilated_unet = dilated_unet(n_filter=64, activation='elu', loss=dice_loss, d_rate=0.2)
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4),
                 ModelCheckpoint(os.path.join(path_model, 'weights_d64.h5'), monitor='val_loss', save_best_only=True,
                                 verbose=1)]
    history_dilated_unet = model_dilated_unet.fit_generator(generator=train_set, steps_per_epoch=TRAIN_STEP,
                                                            validation_data=test_set, validation_steps=VAL_STEP,
                                                            epochs=EPOCH, callbacks=callbacks)

    print("Successfully finished...")
