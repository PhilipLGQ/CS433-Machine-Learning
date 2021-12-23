# Generate submission result with pretrained models
import os
import tensorflow as tf
from keras.models import load_model
from metric_loss import f1, dice_loss
from dataset_UNet import sub_generator, sub_predict_save
from mask_to_submission import generate_submission

# Set random seed
tf.random.set_seed(seed=123)

# Number of submission images
SUB_SIZE = 50

# Path of submission images
path_test = 'data/test_set_images'
path_pred = 'data/prediction'
path_sub = 'submission'
path_model = 'weights'

# List of pretrained models
model_set = {'weights_u32.h5', 'weights_u64.h5', 'weights_d64.h5'}


if __name__ == '__main__':
    if not os.path.exists(path_test):
        print("Please extract test images under directory `data/...`")
        raise FileNotFoundError

    print("Check if any model missing...")
    model_infile = set(os.listdir(path_model))
    print(model_set - model_infile)

    if len(list(model_set - model_infile)):
        print("Some models not found under /weights, please check and retrain missing ones.")
        raise FileNotFoundError

    ensemble = 0

    # Generate ensemble U-Net result directly if False
    print("Load models and generate submission csv...")
    for model_name in model_set:
        model = load_model(os.path.join(path_model, model_name), custom_objects={"f1": f1,
                                                                                 "dice_loss": dice_loss})
        test_imgs = sub_generator(path_test, SUB_SIZE)
        ensemble = ensemble + model.predict_generator(test_imgs, SUB_SIZE, verbose=1)

    # Save ensemble prediction images and create submission csv
    ensemble = ensemble / len(list(model_set))

    sub_predict_save(path_pred, ensemble)
    generate_submission(path_pred, size=SUB_SIZE, csv_filename=os.path.join(path_sub, "submission.csv"))
    print("Successfully finished...")
