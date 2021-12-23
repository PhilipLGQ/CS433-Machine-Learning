import os
import random
import cv2
import numpy as np
from tqdm import tqdm


def is_image_file(filename):
    '''
        filter out files which are not of common image types
    '''
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif", ".tiff"])


# def is_image_damaged(cvimage):
#     '''
#         check image is damaged or not
#         This function is not applied
#     '''
#     height, weight, channel = cvimage.shape
#
#     one_channel = np.sum(cvimage, axis=2)
#     white_pixel_count = len(one_channel[one_channel == 255 * 3])  # Count the number of white pixels
#     if white_pixel_count > 0.08 * height * weight:
#         return True
#     return False


def gamma_transform(img, gamma):
    '''
        conduct gamma transfomation
    '''
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    '''
        conduct random gamma transfomation
    '''
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    '''
        angle: rotation angle
    '''
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb, yb


def blur(img):
    '''
        use cv2 to blur image
    '''
    img = cv2.blur(img, (3, 3))
    return img


def add_noise(img):
    '''
        add random noise to image
    '''
    for i in range(200):
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def data_augment(xb, yb):
    '''
        xb, yb: shape of images
        use random number to decide rotation how much, add noise or not , flip or not, and transform or not
    '''
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 90)
    elif np.random.random() < 0.5:
        xb, yb = rotate(xb, yb, 180)
    elif np.random.random() < 0.75:
        xb, yb = rotate(xb, yb, 270)
    elif np.random.random() < 0.1:
        xb = cv2.flip(xb, 1)
        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb, 1.0)

    elif np.random.random() < 0.5:
        xb = blur(xb)

    elif np.random.random() < 0.75:
        xb = add_noise(xb)

    return xb, yb


def creat_dataset(src_path, label_path, image_sets, img_w, img_h, image_num=1000, mode='augment'):
    '''
        src_path: path of the provided 400*400 image
        label_path: path of the generated training set
        image_sets: a list of files or images under src_path address
        img_w, img_h: shape of image
        image_num: the total number of images that will be generated
        mode: normal or augment; if augment, noise, transform, rotation and other augmentation methods will be applied
    '''
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 1
    for i in tqdm(range(len(image_sets))):
        count = 0

        src_img = cv2.imread(src_path + '/images/' + image_sets[i])
        label_img_gray = cv2.imread(label_path + '/groundtruth/' + image_sets[i].replace('tiff', 'tif'),
                                    cv2.IMREAD_GRAYSCALE)
        X_height, X_width, _ = src_img.shape
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w, :]

            label_roi_gray = label_img_gray[random_height: random_height + img_h, random_width: random_width + img_w]
            if mode == 'augment':
                src_roi, label_roi_gray = data_augment(src_roi, label_roi_gray)
            cv2.imwrite((src_path + '/src/%.3d.png' % g_count), src_roi)
            cv2.imwrite((label_path + '/label/%.3d.png' % g_count), label_roi_gray)
            count += 1
            g_count += 1


def creat_val_set(src_path, label_path, image_sets, img_w, img_h, image_num=160, mode='augment'):
    '''
        src_path: path of the provided 400*400 image
        label_path: path of the generated validation set
        image_sets: a list of files or images under src_path address
        img_w, img_h: shape of image
        image_num: the total number of images that will be generated
        mode: normal or augment; if augment, noise, transform, rotation and other augmentation methods will be applied
    '''
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 1
    for i in range(0,len(image_sets)):
        count = 0
        if g_count == image_num+1:
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            break
        src_img = cv2.imread(src_path + '/images/' + image_sets[i])
        label_img_gray = cv2.imread(label_path + '/groundtruth/' + image_sets[i].replace('tiff', 'tif'),
                                    cv2.IMREAD_GRAYSCALE)
        X_height, X_width, _ = src_img.shape
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w, :]

            label_roi_gray = label_img_gray[random_height: random_height + img_h, random_width: random_width + img_w]
            if mode == 'augment':
                src_roi, label_roi_gray = data_augment(src_roi, label_roi_gray)
            cv2.imwrite(('data_SegNet/validation' + '/src/%.3d.png' % g_count), src_roi)
            cv2.imwrite(('data_SegNet/validation' + '/label/%.3d.png' % g_count), label_roi_gray)
            count += 1
            g_count += 1


if __name__ == '__main__':
    img_w = 320
    img_h = 320
    src_data_path = 'data_SegNet/training'
    label_data_path = 'data_segnet/training'
    image_sets2 = [x for x in os.listdir(src_data_path+'/images') if is_image_file(x)]
    creat_dataset(src_path=src_data_path, label_path=label_data_path, image_sets=image_sets2, img_w=img_w, img_h=img_h)
    creat_val_set(src_path=src_data_path, label_path=label_data_path, image_sets=image_sets2, img_w=img_w, img_h=img_h)
