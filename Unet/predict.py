import os
import nibabel as nb
from argparse import ArgumentParser
from image_processing import create_patches_from_images, recreate_image_from_patches
from keras.models import load_model
from metrics import (dice_coefficient, dice_coefficient_loss, dice_coef, dice_coef_loss,
                            weighted_dice_coefficient_loss, weighted_dice_coefficient)


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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to .nii image")
    parser.add_argument("-m", "--mask", help="Path to .nii image")

    args = parser.parse_args()

    image = nb.load(args.input).get_data()
    mask = nb.load(args.mask).get_data()
    patches = create_patches_from_images(image, [32, 32, 32])

    model = load_old_model("/home/Simon/Datasets/model.25--0.56.hdf5")

    predicted = []
    for p in patches:
        predicted.append(model.predict(p))

    y_pred = recreate_image_from_patches(image.shape, predicted)
    new_filename = "predicted_{}"+os.path.basename(args.input)

    old_mask_filename = args.mask
    new_mask_filename = old_mask_filename.replace(os.path.basename(args.mask), new_filename)

    nb.save(y_pred, new_mask_filename)

    print(dice_coefficient_loss(mask,y_pred))