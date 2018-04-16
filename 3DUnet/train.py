import keras

from keras import Sequential
from keras.layers import Conv3D
from keras.callbacks import TensorBoard
from argparse import ArgumentParser
import subprocess
import os


def create_if_not_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    return dir

def create_patches (data):
    return data


def train(model, training_data, validation_data, logdir=None):

    tensorboard_callback = None
    if logdir is not None:
        log_path = create_if_not_exists(os.path.join(logdir,"logs"))
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

    train_x, train_y = zip(*training_data)

    # Train the model, iterating on the data in batches of 32 samples
    history = model.fit(x=train_x, y=train_y, validation_data=validation_data, steps_per_epoch=steps_per_epoch,
                        validation_steps = validation_steps, shuffle=shuffle, epochs=10000, batch_size=32,
                        callbacks=[tensorboard_callback])

    return model, history


if __name__ == '__main__':
    parser = ArgumentParser(description="Train a 3D Model unet")
    parser.add_argument("-l","--logdir", help="Directory where to log the data for tensorboard",
                        default ="/home/snarduzz/")

   