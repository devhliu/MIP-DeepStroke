import os
import nibabel as nb
from argparse import ArgumentParser
from image_processing import create_2D_patches_from_images, recreate_image_from_2D_patches, preprocess_image
from keras.models import load_model
from metrics import (dice_coefficient, dice_coefficient_loss, dice_coef, dice_coef_loss,
                     weighted_dice_coefficient_loss, weighted_dice_coefficient, tversky_loss, tversky_coeff)
import numpy as np
from utils2D import channel_first_to_channel_last


def load_old_model(model_file):
    print("Loading pre-trained model")
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 'dice_coefficient': dice_coefficient,
                      'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss,
                      'weighted_dice_coefficient': weighted_dice_coefficient,
                      'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss,
                      'tversky_loss': tversky_loss,
                      'tversky_coeff': tversky_coeff}
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


def predict_patch(patches, model):
    patch_extended = patches.reshape(1, patches.shape[0], patches.shape[1], patches.shape[2], patches.shape[3])
    return model.predict(patch_extended)[0, 0, :, :, :]


def predict(images, model, patch_size, verbose=0, batch_size=32):
    if not isinstance(images, list):
        images = [images]  # Transform to 1 dimensional channel

    original_image_size = images[0].shape

    patch_by_channel = []
    for i in range(len(images)):
        image = images[i]
        if len(image.shape) > 3:
            print("Found {} 3D channels in prediction image {}. Taking first channel.".format(len(image.shape), i))
            image = image[:, :, :, 0]
            original_image_size = image.shape
        image_norm = preprocess_image(image)
        image_patches = create_2D_patches_from_images(image_norm, patch_size)
        patch_by_channel.append(image_patches)

    # at that point, patch_by_channel shape = [channels, num_patches, dimx, dimy]
    patch_by_channel = np.array(patch_by_channel)
    shape = patch_by_channel.shape
    patch_channel_tensor = np.zeros([shape[1], shape[2], shape[3], shape[0]])

    for i in range(shape[0]):
        patch_channel_tensor[:, :, :, i] = patch_by_channel[i, :, :, :]

    try:
        predictions = model.predict(np.asarray(patch_channel_tensor), batch_size=batch_size, verbose=verbose)
    except:  # Not enough memory : switch to mono prediction
        predictions = model.predict(np.asarray(patch_channel_tensor), batch_size=1, verbose=verbose)


    dimensions = len(np.array(model.output_shape).shape)
    if dimensions > 1:
        model_output_depth = len(model.output_shape)
    else:
        model_output_depth = 1

    if model_output_depth == 1:
        predictions = predictions[:, :, :, 0]
    else:
        predictions = predictions[model_output_depth-1][:, :, :, 0]  # last output has the highest definition

    # reshape to [(dimx,dimy),dimz]
    list_patches = []
    for i in range(predictions.shape[0]):
        list_patches.append(predictions[i, :, :])

    predicted_image = recreate_image_from_2D_patches(original_image_size=original_image_size, list_patches=list_patches)
    return predicted_image


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to .nii image")
    parser.add_argument("-s", "--mask", help="Path to save .nii image")
    parser.add_argument("-m", "--model", help="Path to model file or weights")

    args = parser.parse_args()

    image = nb.load(args.input).get_data()
    mask = nb.load(args.mask).get_data()
    model = load_old_model(args.model)
    patches = create_2D_patches_from_images(image, [512, 512])

    predicted = []
    for p in patches:
        predicted.append(model.predict(p))

    # image.shape = (197, 233, 189)
    y_pred = recreate_image_from_2D_patches(image.shape, predicted)
    new_filename = "predicted_{}" + os.path.basename(args.input)

    old_mask_filename = args.mask
    new_mask_filename = old_mask_filename.replace(os.path.basename(args.mask), new_filename)

    nb.save(y_pred, new_mask_filename)

    print(dice_coefficient_loss(mask, y_pred))
