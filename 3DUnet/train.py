import keras

from keras import Sequential
from keras.layers import Conv3D
from keras.callbacks import TensorBoard
from argparse import ArgumentParser
import subprocess
import os
import nibabel as nb
import numpy as np
import progressbar
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from .utils import create_if_not_exists
from .image_processing import transform_atlas_to_patches


def create_generators(batch_size, data_path=None):
    train_path = "/train/"
    validation_path = "validation/"
    if data_path is not None:
        train_path = os.path.join(data_path,train_path)
        validation_path = os.path.join(data_path, validation_path)

    train_generator = zip(generator(os.path.join(train_path, "inputs"), batch_size=batch_size),
                          (generator(os.path.join(train_path, "masks"), batch_size=batch_size)))

    validation_generator = zip(generator(os.path.join(validation_path, "inputs"), batch_size=batch_size),
                          (generator(os.path.join(validation_path, "masks"), batch_size=batch_size)))

    return train_generator, validation_generator


def generator(from_directory, batch_size):
    while True:
        image_path = os.listdir(from_directory)
        for cbatch in range(0, len(image_path), batch_size):
            images = np.array([nb.load(path).get_data() for path in image_path[cbatch:(cbatch + batch_size)]])
            yield images


def train(model, training_data, validation_data, logdir=None):

    tensorboard_callback = None
    if logdir is not None:
        log_path = create_if_not_exists(os.path.join(logdir, "logs"))
        tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=0, batch_size=32, write_graph=True,
                                           write_grads=True, write_images=True, embeddings_freq=0,
                                           embeddings_layer_names=None, embeddings_metadata=None)
        # Start Tensorboard
        subprocess.run(["tensorboard --logdir={}".format(log_path)])

    # Save checkpoint each 5 epochs
    checkpoint_path = create_if_not_exists(os.path.join(logdir,"checkpoints"))

    keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=False,
                                    save_weights_only=False, mode='auto', period=10)

    # Parameters
    validation_steps = 5   # Number of steps per evaluation (number of to pass)
    steps_per_epoch = 20   # Number of batches to pass before going to next epoch
    shuffle = True         # Shuffle the data before creating a batch
    batch_size = 32

    train_x, train_y = zip(*training_data)

    # Train the model, iterating on the data in batches of 32 samples
    history = model.fit(x=train_x, y=train_y, validation_data=validation_data, steps_per_epoch=len(train_x)/batch_size,
                        validation_steps = validation_steps, shuffle=shuffle, epochs=10000, batch_size=batch_size,
                        callbacks=[tensorboard_callback])

    return model, history