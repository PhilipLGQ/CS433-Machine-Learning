import numpy as np
import tensorflow as tf
import os
import skimage.io as io
import random
import glob
import shutil

from PIL import Image
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

# Set random seed
tf.random.set_seed(seed=123)


# Create directory if it not exists
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Preprocess functions for images and masks
def img_preprocess(img):
    output = img / 255
    return output


def mask_preprocess(mask):
    output = mask / 255

    # make clear of boundaries
    output[output >= 0.5] = 1
    output[output < 0.5] = 0

    return output


# Image augmentation for both training set & validation set
def data_aug(path_tr, path_te, target_size=(320, 320), original_size=(400, 400), test_size=0.2, seed=123):
    """
        Arguments:
            path_tr: path of training set
            path_te: path of test set
            test_size: ratio between test set & training set
            seed: random seed, set to 123 by default
    """

    # augment by left-right flip and multi-angle rotations
    for i in range(1, 101):
        # original images
        img = np.asarray(Image.open(os.path.join(path_tr, 'images', 'satImage_%.3d.png' % i)))
        mask = np.asarray(Image.open(os.path.join(path_tr, 'groundtruth', 'satImage_%.3d.png' % i)))

        random.seed(seed)
        random_width = random.randint(0, original_size[1] - target_size[1] - 1)
        random.seed(seed)
        random_height = random.randint(0, original_size[0] - target_size[0] - 1)

        img = Image.fromarray(img[random_height: random_height + target_size[0], random_width: random_width + target_size[1], :])
        mask = Image.fromarray(mask[random_height: random_height + target_size[0], random_width: random_width + target_size[1]])

        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        io.imsave(os.path.join(path_tr, 'images', 'satImage_%.3d.png' % i), np.array(img))
        io.imsave(os.path.join(path_tr, 'images', 'satImage_%.3d_f.png' % i), np.array(img_flip))

        # groundtruth masks

        mask_flip = mask.transpose(Image.FLIP_LEFT_RIGHT)
        io.imsave(os.path.join(path_tr, 'groundtruth', 'satImage_%.3d_f.png' % i), np.array(mask_flip))
        io.imsave(os.path.join(path_tr, 'groundtruth', 'satImage_%.3d.png' % i), np.array(mask))

        # rotations
        for rotation in [90, 180, 270]:
            img_rotation = img.rotate(rotation)
            io.imsave(os.path.join(path_tr, 'images', 'satImage_%.3d_%.3d.png' % (i, rotation)),
                      np.array(img_rotation))

            img_flip_rotation = img_flip.rotate(rotation)
            io.imsave(os.path.join(path_tr, 'images', 'satImage_%.3d_f_%.3d.png' % (i, rotation)),
                      np.array(img_flip_rotation))

            mask_rotation = mask.rotate(rotation)
            io.imsave(os.path.join(path_tr, 'groundtruth', 'satImage_%.3d_%.3d.png' % (i, rotation)),
                      np.array(mask_rotation))

            mask_flip_rotation = mask_flip.rotate(rotation)
            io.imsave(os.path.join(path_tr, 'groundtruth', 'satImage_%.3d_f_%.3d.png' % (i, rotation)),
                      np.array(mask_flip_rotation))

    print("Data augmentation finished...")
    print("Splitting training set & test set...")

    # train-test split
    data_aug = os.listdir(os.path.join(path_tr, 'images'))
    img_tr, img_te = train_test_split(data_aug, test_size=test_size, random_state=seed)

    # save test split to new folders
    create_dir(path_te)
    create_dir(os.path.join(path_te, 'images'))
    create_dir(os.path.join(path_te, 'groundtruth'))
    for img in img_te:
        os.rename(os.path.join(path_tr, 'images', img), os.path.join(path_te, 'images', img))
        os.rename(os.path.join(path_tr, 'groundtruth', img), os.path.join(path_te, 'groundtruth', img))


# Generate training set and test set from directory
def set_generator(batch_size, aug_dict, path_tr, path_te, image_folder='images', mask_folder='groundtruth',
                  dir_tr=None, dir_te=None, target_size=(320, 320), seed=123):
    """
        Arguments:
            batch_size: # images each batch
            aug_dict: data augmentation parameters for ImageDataGenerator
            path_tr: path of the training set
            path_te: path of the validation set
            image_folder: image folder's name
            mask_folder: mask folder's name
            dir_tr: save training set to dir if not None
            dir_te: save test set to dir if not None
            target_size: size of target images, by default 400 * 400 (img size)
            seed: random seed, set to 123 by default

        Returns:
            (trainGen, valGen): generators for training & test set
    """

    # training set generator
    if dir_tr:
        create_dir(dir_tr)

    # add normalization preprocessing function to dict
    aug_parameter_dict = aug_dict.copy()
    aug_parameter_dict["preprocessing_function"] = img_preprocess

    # further augmentation by ImageDataGenerator
    img_generator = ImageDataGenerator(**aug_parameter_dict)

    train_generator = img_generator.flow_from_directory(path_tr, classes=[image_folder],
                                                        class_mode=None,
                                                        color_mode="rgb",
                                                        target_size=target_size,
                                                        batch_size=batch_size,
                                                        save_to_dir=dir_tr,
                                                        save_prefix="image",
                                                        seed=seed)

    # do the same to masks correspondingly
    # mask_dict = aug_dict.copy()
    # mask_dict["preprocessing_function"] = preprocess_mask
    aug_parameter_mask = aug_dict.copy()
    aug_parameter_mask["preprocessing_function"] = mask_preprocess
    mask_generator = ImageDataGenerator(**aug_parameter_mask)

    train_mask_generator = mask_generator.flow_from_directory(path_tr, classes=[mask_folder],
                                                              class_mode=None,
                                                              color_mode="grayscale",
                                                              target_size=target_size,
                                                              batch_size=batch_size,
                                                              save_to_dir=dir_tr,
                                                              save_prefix="mask",
                                                              seed=seed)

    # test set generator
    if dir_te:
        create_dir(dir_te)

    img_generator = ImageDataGenerator(preprocessing_function=img_preprocess)

    test_generator = img_generator.flow_from_directory(path_te, classes=[image_folder],
                                                       class_mode=None,
                                                       color_mode="rgb",
                                                       target_size=target_size,
                                                       batch_size=batch_size,
                                                       save_to_dir=dir_te,
                                                       save_prefix="image",
                                                       seed=seed,
                                                       shuffle=False)

    mask_generator = ImageDataGenerator(preprocessing_function=mask_preprocess)

    test_mask_generator = mask_generator.flow_from_directory(path_te, classes=[mask_folder],
                                                             class_mode=None,
                                                             color_mode="grayscale",
                                                             target_size=target_size,
                                                             batch_size=batch_size,
                                                             save_to_dir=dir_te,
                                                             save_prefix="mask",
                                                             seed=seed,
                                                             shuffle=False)

    return zip(train_generator, train_mask_generator), zip(test_generator, test_mask_generator)


# Regenerate training & test set from Ensemble UNet predictions
def set_regenerator(path, set):
    # collect images in training set
    for f_name in glob.glob(os.path.join(path, set, '*.png')):
        img = io.imread(f_name)

        # Normalize pixels to [0, 1]
        img = img / 255
        img = np.reshape(img, (1,) + img.shape)

        yield img


# Copy groundtruth masks to data_segnet directory
def mask_copy(path_tr, path_te, path_segnet):
    shutil.copytree(os.path.join(path_tr, 'groundtruth'), os.path.join(path_segnet, 'training/groundtruth'))
    shutil.copytree(os.path.join(path_te, 'groundtruth'), os.path.join(path_segnet, 'validation/groundtruth'))


# Save ensembled masks from regenerated training & test set
def ensemble_generator(path):
    for img_name in os.listdir(os.path.join(path, 'images')):
        img = io.imread(os.path.join(path, 'images', img_name))

        # Normalize pixels to [0, 1]
        img = img / 255
        img = np.reshape(img, (1,) + img.shape)

        yield img


# Save predicted ensemble masks from training and test set
def mask_save(path_segnet, set, numpy_file):
    for idx, array_img in enumerate(numpy_file):
        img = array_img[:, :, 0]
        io.imsave(os.path.join(path_segnet, set, 'mask', '%.3d.png' % (idx + 1)), img)


# Save final prediction images to named path
def sub_predict_save(path, numpy_file):
    """
        Arguments:
            path: path to save the predictions
            numpy_file: numpy array of predicted submission images
    """
    create_dir(path)

    for idx, array_img in enumerate(numpy_file):
        img = array_img[:, :, 0]
        io.imsave(os.path.join(path, '%.3d.png' % (idx + 1)), img)


# Generator for submission set
def sub_generator(path_sub, n_imgs=50):
    """
        Arguments:
            path_sub: path of submission set
            n_imgs: # images of submission test, by default 50
    """
    for i in range(1, n_imgs + 1):
        img = io.imread(os.path.join(path_sub, "test_%d" % i, "test_%d.png" % i))

        # Normalize pixels to [0, 1]
        img = img / 255
        img = np.reshape(img, (1,) + img.shape)

        yield img
