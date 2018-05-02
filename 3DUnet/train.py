import keras

from keras.callbacks import TensorBoard
import subprocess
import os
import nibabel as nb
import numpy as np
from utils import create_if_not_exists, TrainValTensorBoard
from unet import unet_model_3d
from argparse import ArgumentParser
import time
import tensorflow as tf
from tensorflow.python.client import device_lib
import keras

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 6} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

def create_generators(batch_size, data_path=None, skip_blank=True):
    train_path = "train/"
    validation_path = "validation/"
    if data_path is not None:
        train_path = os.path.join(data_path, train_path)
        validation_path = os.path.join(data_path, validation_path)

    print("Train data path {}".format(train_path))
    print("Validation data path {}".format(validation_path))

    # train_generator = zip(generator(os.path.join(train_path, "inputs"), batch_size=batch_size, skip_blank=skip_blank),
    # (generator(os.path.join(train_path, "masks"), batch_size=batch_size, skip_blank=skip_blank)))

    # validation_generator = zip(generator(os.path.join(validation_path, "inputs"), batch_size=batch_size,
    # skip_blank=skip_blank),
    # (generator(os.path.join(validation_path, "masks"), batch_size=batch_size, skip_blank=skip_blank)))

    train_generator = dual_generator(os.path.join(train_path, "inputs"), os.path.join(train_path, "masks"),
                                     batch_size=batch_size, skip_blank=skip_blank)

    validation_generator = dual_generator(os.path.join(validation_path, "inputs"), os.path.join(validation_path, "masks"),
                                          batch_size=batch_size, skip_blank=skip_blank)

    return train_generator, validation_generator


def generator(from_directory, batch_size, skip_blank=True):
    while True:
        image_path = os.listdir(from_directory)
        for cbatch in range(0, len(image_path), batch_size):
            images = []
            for path in image_path[cbatch:(cbatch + batch_size)]:

                image = nb.load(os.path.join(from_directory, path)).get_data().reshape(1,32,32,32)
                if not (np.all(image == 0) and skip_blank):
                    images.append(image)

            images = np.array(images)

            yield images


def dual_generator(input_directory, target_directory, batch_size, skip_blank=True):
    while True:
        image_paths = os.listdir(input_directory)
        target_paths = os.listdir(target_directory)
        paths = list(zip(image_paths, target_paths))

        for cbatch in range(0, len(paths), batch_size):
            x_list = []
            y_list = []
            for input_path, target_path in paths[cbatch:(cbatch + batch_size)]:
                image = nb.load(os.path.join(input_directory, input_path)).get_data()
                target = nb.load(os.path.join(target_directory, target_path)).get_data()

                shape = image.shape
                reshape_size = (1, shape[0], shape[1], shape[2])
                image = image.reshape(reshape_size)
                target = target.reshape(reshape_size)

                if not (np.all(image == 0) and skip_blank):
                    x_list.append(image)
                    y_list.append(target)

            x_list = np.array(x_list)
            y_list = np.array(y_list)

            yield x_list, y_list


def train(model, data_path, batch_size=32, logdir=None, skip_blank=True):

    tensorboard_callback = None
    if logdir is not None:
        log_path = create_if_not_exists(os.path.join(logdir, "logs"))
        tensorboard_callback = TrainValTensorBoard(log_dir=log_path, histogram_freq=0, batch_size=batch_size,
                                                   write_graph=True,
                                           write_grads=True, write_images=True, embeddings_freq=0,
                                           embeddings_layer_names=None, embeddings_metadata=None)
        # Start Tensorboard
        print("tensorboard --logdir={}".format(log_path))

    # Save checkpoint each 5 epochs
    checkpoint_path = create_if_not_exists(os.path.join(logdir, "checkpoints"))
    checkpoint_filename = os.path.join(checkpoint_path,"model.{epoch:02d}-{val_loss:.2f}.hdf5")

    checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_filename, monitor='val_loss', verbose=0, save_best_only=False,
                                    save_weights_only=False, mode='auto', period=5)

    # Parameters
    validation_steps = 5   # Number of steps per evaluation (number of to pass)
    steps_per_epoch = 1000  # Number of batches to pass before going to next epoch
    shuffle = True         # Shuffle the data before creating a batch

    training_generator, validation_generator = create_generators(batch_size, data_path=data_path, skip_blank=skip_blank)

    # Train the model, iterating on the data in batches of 32 samples
    history = model.fit_generator(training_generator, steps_per_epoch=steps_per_epoch, epochs=500, verbose=1,
                                  callbacks=[tensorboard_callback, checkpoint_callback],
                                  validation_data=validation_generator, validation_steps=validation_steps,
                                  class_weight=None, max_queue_size=10,
                                  workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

    return model, history


if __name__ == '__main__':
    parser = ArgumentParser(description="Train a 3D Model unet")
    parser.add_argument("-l", "--logdir", help="Directory where to log the data for tensorboard",
                        default="/home/simon/")

    parser.add_argument("-d", "--data_path", help="Path to data folder",
                        default="/home/simon/Datasets/Data/32x32x32")
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument("-s", "--skip_blank", help="Skip blank images - will not be fed to the network", default=True)

    args = parser.parse_args()
    data_path = args.data_path
    logdir = os.path.join(args.logdir, time.strftime("%Y%m%d_%H-%M-%S", time.gmtime()))
    batch_size = args.batch_size

    create_if_not_exists(logdir)
    model = unet_model_3d([1, 32, 32, 32], batch_normalization=True)
    train(model, batch_size=batch_size, data_path=data_path, logdir=logdir, skip_blank=args.skip_blank)