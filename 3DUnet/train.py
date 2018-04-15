import keras

from keras import Sequential
from keras.layers import Conv3D
from argparse import ArgumentParser


def train(model, training_data, validation_data, logdir=None):



if __name__ == '__main__':
    parser = ArgumentParser(description="Train a 3D Model unet")
    parser.add_argument("-l","--logdir", help="Directory where to log the data for tensorboard",
                        default ="/home/snarduzz/")

   