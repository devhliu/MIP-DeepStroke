import sys
sys.path.append('../')
from argparse import ArgumentParser
from keras.models import load_model
from UnetCT.metrics import *

if __name__ == '__main__':
    parser = ArgumentParser(description="Train a 3D Model unet")
    parser.add_argument("-m", "--model_name", help="Directory where the Keras models are stored",
                        default="/home/simon/models")

    args = parser.parse_args()

    ckpt = args.model_name

    model = load_model(ckpt, custom_objects={"dice_coefficient" : dice_coefficient,
                                            "dice_coefficient_loss" : dice_coefficient_loss,
                                            "weighted_dice_coefficient_loss" : weighted_dice_coefficient_loss,
                                            "weighted_dice_coefficient" : weighted_dice_coefficient,
                                            "tversky_loss" : tversky_loss,
                                            "tversky_coeff": tversky_coeff
                                            })
    print("----Parameters----")
    print(model.count_params())
    print()
    print("----Summary----")
    print(model.summary())
