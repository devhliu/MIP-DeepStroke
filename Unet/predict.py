import os
import nibabel as nb
from argparse import ArgumentParser
from Unet.image_processing import create_patches_from_images, recreate_image_from_patches, preprocess_image
from keras.models import load_model
from .metrics import (dice_coefficient, dice_coefficient_loss, dice_coef, dice_coef_loss,
                            weighted_dice_coefficient_loss, weighted_dice_coefficient,tversky_loss,tversky_coeff)
import numpy as np


def load_old_model(model_file):
    print("Loading pre-trained model")
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 'dice_coefficient': dice_coefficient,
                      'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss,
                      'weighted_dice_coefficient': weighted_dice_coefficient,
                      'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss}
    try:
        from keras_contrib.layers import InstanceNormalization
        custom_objects["InstanceNormalization"] = InstanceNormalization
    except ImportError:
        pass
    try:
        return load_model(model_file, custom_objects=custom_objects)
    except ValueError as error:
        if 'InstanceNormalization' in str(error):
            raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                          "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
        else:
            raise error


def predict_patch(patch, model):
    patch_extended = patch.reshape(1, 1, patch.shape[0], patch.shape[1], patch.shape[2])
    return model.predict(patch_extended)[0, 0, :, :, :]


def predict(image, model, patch_size, verbose=0):
    original_image_size = image.shape
    image_norm = preprocess_image(image)
    image_patches = create_patches_from_images(image_norm, patch_size)
    # Extend with channel
    image_patches = [x.reshape(1, patch_size[0], patch_size[1], patch_size[2]) for x in image_patches]
    try:
        predictions = model.predict(np.asarray(image_patches), batch_size=32, verbose=verbose)[:, 0, :, :, :]
    except: # Not enough memory : switch to mono prediction
        predictions = model.predict(np.asarray(image_patches), batch_size=1, verbose=verbose)[:, 0, :, :, :]

    predicted_image = recreate_image_from_patches(original_image_size=original_image_size, list_patches=predictions)
    return predicted_image


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path", help="Path to .nii image")
    parser.add_argument("-m", "--model_path", help="Path to model")
    parser.add_argument("-s", "--save_folder", help="Path to save the image")
    parser.add_argument("-p", "--patch_size", type=int, help="Patch size", default=32)

    args = parser.parse_args()

    input_path = args.input_path
    model_path = args.model
    save_folder = args.save_folder
    p = args.patch_size

    img_input = nb.load(input_path)
    image = img_input.get_data()
    patch_size = [p, p, p]

    model = load_old_model(model_path, custom_objects={"dice_coefficient" : dice_coefficient,
                                                        "dice_coefficient_loss" : dice_coefficient_loss,
                                                        "weighted_dice_coefficient_loss" : weighted_dice_coefficient_loss,
                                                        "weighted_dice_coefficient" : weighted_dice_coefficient,
                                                        "tversky_loss" : tversky_loss,
                                                        "tversky_coeff": tversky_coeff
                                                        })

    image_pred = predict(image, model, patch_size)

    image_extension = '.nii'
    filename_save = os.path.join(save_folder, "predicted-{}".format(os.path.basename(input_path)))
    if not filename_save.endswith(image_extension):
        filename_save = filename_save+image_extension

    coordinate_space = img_input.affine
    predicted_img = nb.Nifti1Image(image_pred, affine=coordinate_space)
    nb.save(predicted_img, filename_save)

