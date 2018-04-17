import keras

from keras.callbacks import TensorBoard
import subprocess
import os
import nibabel as nb
import numpy as np
from utils import create_if_not_exists
from unet import unet_model_3d


def create_generators(batch_size, data_path=None):
    train_path = "train/"
    validation_path = "validation/"
    if data_path is not None:
        train_path = os.path.join(data_path, train_path)
        validation_path = os.path.join(data_path, validation_path)

    print("Train data path {}".format(train_path))
    print("Validation data path {}".format(validation_path))

    train_generator = zip(generator(os.path.join(train_path, "inputs"), batch_size=batch_size),
                          (generator(os.path.join(train_path, "masks"), batch_size=batch_size)))

    validation_generator = zip(generator(os.path.join(validation_path, "inputs"), batch_size=batch_size),
                          (generator(os.path.join(validation_path, "masks"), batch_size=batch_size)))

    return train_generator, validation_generator


def generator(from_directory, batch_size):
    while True:
        image_path = os.listdir(from_directory)
        for cbatch in range(0, len(image_path), batch_size):
            images = np.array([
                # TODO FIX
                nb.load(os.path.join(from_directory, path)).get_data().reshape(1,32,32,32)



                               for path in image_path[cbatch:(cbatch + batch_size)]])

            yield images


def train(model, data_path, batch_size=32, logdir=None):

    tensorboard_callback = None
    if logdir is not None:
        log_path = create_if_not_exists(os.path.join(logdir, "logs"))
        tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=0, batch_size=batch_size, write_graph=True,
                                           write_grads=True, write_images=True, embeddings_freq=0,
                                           embeddings_layer_names=None, embeddings_metadata=None)
        # Start Tensorboard
        print("tensorboard --logdir={}".format(log_path))

    # Save checkpoint each 5 epochs
    checkpoint_path = create_if_not_exists(os.path.join(logdir, "checkpoints"))

    keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=False,
                                    save_weights_only=False, mode='auto', period=10)

    # Parameters
    validation_steps = 5   # Number of steps per evaluation (number of to pass)
    steps_per_epoch = 1   # Number of batches to pass before going to next epoch
    shuffle = True         # Shuffle the data before creating a batch

    training_generator, validation_generator = create_generators(batch_size, data_path=data_path)

    # Train the model, iterating on the data in batches of 32 samples
    history = model.fit_generator(training_generator, steps_per_epoch=steps_per_epoch, epochs=100, verbose=1,
                                  callbacks=[tensorboard_callback],
                                  validation_data=validation_generator, validation_steps=validation_steps,
                                  class_weight=None, max_queue_size=10,
                                  workers=1, use_multiprocessing=True, shuffle=True, initial_epoch=0)

    return model, history

if __name__ == '__main__':
    data_path = "/home/simon/Datasets/Data/32x32x32/"
    logdir = os.path.join(data_path, "logdir")
    create_if_not_exists(logdir)
    model = unet_model_3d([1, 32, 32, 32], batch_normalization=True)
    batch_size=3
    train(model, batch_size=batch_size, data_path=data_path, logdir=logdir)