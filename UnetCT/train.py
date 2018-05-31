import keras

from keras.callbacks import TensorBoard
import subprocess
import os
import nibabel as nb
import numpy as np
from utils import create_if_not_exists
from unet import unet_model_3d
from callbacks import TrainValTensorBoard
from argparse import ArgumentParser
import time
import tensorflow as tf
from tensorflow.python.client import device_lib
import keras
from metrics import dice_coefficient, weighted_dice_coefficient, weighted_dice_coefficient_loss, dice_coefficient_loss, tversky_coeff, tversky_loss
from keras.metrics import binary_crossentropy, binary_accuracy
from create_datasets import load_data_for_patient
import tensorflow as tf

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 6} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)


def create_generators(batch_size, data_path=None, skip_blank=True, folders_input=['input'], folders_target=['lesion']):
    train_path = "train/"
    validation_path = "validation/"
    if data_path is not None:
        train_path = os.path.join(data_path, train_path)
        validation_path = os.path.join(data_path, validation_path)

    training_size = len(os.listdir(os.path.join(train_path, "MTT")))
    validation_size = len(os.listdir(os.path.join(validation_path, "MTT")))

    print("Train data path {} - {} samples".format(train_path, training_size))
    print("Validation data path {} - {} samples".format(validation_path, validation_size))

    train_generator = dual_generator(train_path, folders_input, folders_target, batch_size=batch_size, skip_blank=skip_blank)

    validation_generator = dual_generator(validation_path, folders_input, folders_target, batch_size=batch_size, skip_blank=skip_blank)

    return train_generator, validation_generator


def dual_generator(data_directory, folders_input, folders_target, batch_size, skip_blank=False, logfile=None):
    while True:
        example_dir = os.path.join(data_directory, folders_input[0])
        image_paths = os.listdir(example_dir)

        x_list = []
        y_list = []
        for i in range(len(image_paths)):

            inputs = []
            inputs_paths = []
            for x in folders_input:
                image_path = image_paths[i].replace(folders_input[0], x)
                image_input = nb.load(os.path.join(data_directory, x, image_path)).get_data()
                inputs.append(image_input)
                inputs_paths.append(image_path)

            targets = []
            targets_paths = []
            for y in folders_target:
                target_path = image_paths[i].replace(folders_input[0], y)
                image_target = nb.load(os.path.join(data_directory, y, target_path)).get_data()
                targets.append(image_target)
                targets_paths.append(target_path)
            if logfile:
                with open(logfile, "a") as f:
                    paths = inputs_paths+targets_paths
                    f.write("{} - {}".format(i, paths))

            if not (np.all(inputs == 0) and skip_blank):
                x_list.append(inputs)
                y_list.append(targets)

            if len(x_list) == batch_size:
                x_list_batch = np.array(x_list)
                y_list_batch = np.array(y_list)
                x_list = []
                y_list = []

                yield x_list_batch, y_list_batch

        while len(x_list)<batch_size:
           input_size = nb.load(os.path.join(data_directory, folders_input[0], image_paths[0])).get_data().shape
           zero = np.zeros(input_size)

           inputs = []
           for x in folders_input:
               inputs.append(zero)
           targets = []
           for y in folders_target:
               targets.append(zero)

           x_list.append(inputs)
           y_list.append(targets)
        yield np.array(x_list), np.array(y_list)




def train(model, data_path, batch_size=32, logdir=None, skip_blank=True, epoch_size=None, patch_size=None, folders_input=['input'], folders_target=['lesion'], num_patient=295742):

    training_generator, validation_generator = create_generators(batch_size, data_path=data_path, skip_blank=skip_blank,
                                                                 folders_input=folders_input,
                                                                 folders_target=folders_target)

    dataset_training_size = len(os.listdir(os.path.join(data_path, "train",folders_input[0])))
    dataset_val_size = len(os.listdir(os.path.join(data_path, "validation",folders_input[0])))

    tensorboard_callback = None
    if logdir is not None:
        log_path = create_if_not_exists(os.path.join(logdir, "logs"))

        # load image and lesion
        patient_path = "/home/klug/data/preprocessed_original/{}".format(num_patient)
        MTT, CBF, CBV, Tmax, lesion = load_data_for_patient(patient_path)

        dict_inputs = {"MTT": MTT,
                       "CBF": CBF,
                       "CBV": CBV,
                       "Tmax": Tmax,
                       "lesion":lesion}

        print("Shapes of data : ")
        for k in dict_inputs.keys():
            print("\t", k, " - ", dict_inputs[k].shape)

        images_input = [dict_inputs[k] for k in folders_input]
        images_target = [lesion]
        layer = int(MTT.shape[2]/2.0)
        layers = [layer]
        training_generator_log, validation_generator_log = create_generators(batch_size, data_path=data_path,
                                                                     skip_blank=skip_blank,
                                                                     folders_input=folders_input,
                                                                     folders_target=folders_target)
        tensorboard_callback = TrainValTensorBoard(log_dir=log_path,
                                                   images=images_input,
                                                   lesions=images_target,
                                                   layers=layers,
                                                   patch_size=patch_size,
                                                   training_generator=training_generator_log,
                                                   validation_generator=validation_generator_log,
                                                   validation_steps= int(dataset_val_size/batch_size),
                                                   verbose=1,
                                                   histogram_freq=0,
                                                   batch_size=batch_size,
                                                   write_graph=True,
                                                   write_grads=True,
                                                   write_images=True,
                                                   embeddings_freq=0,
                                                   embeddings_layer_names=None,
                                                   embeddings_metadata=None)

        # Start Tensorboard
        print("tensorboard --logdir={}".format(log_path))

    # Save checkpoint each 5 epochs
    checkpoint_path = create_if_not_exists(os.path.join(logdir, "checkpoints"))
    checkpoint_filename = os.path.join(checkpoint_path, "model.{epoch:02d}-{val_loss:.2f}.hdf5")

    checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_filename, monitor='val_loss', verbose=0, save_best_only=False,
                                    save_weights_only=False, mode='auto', period=1)

    # Parameters
    validation_steps = 1   # Number of steps per evaluation (number of to pass)
    steps_per_epoch = (dataset_training_size/batch_size)  # Number of batches to pass before going to next epoch
    if epoch_size is not None :
        steps_per_epoch = epoch_size

    shuffle = True         # Shuffle the data before creating a batch

    # Train the model, iterating on the data in batches of 32 samples
    history = model.fit_generator(training_generator, steps_per_epoch=steps_per_epoch, epochs=500, verbose=1,
                                  callbacks=[tensorboard_callback, checkpoint_callback],
                                  validation_data=validation_generator, validation_steps=validation_steps,
                                  class_weight=None, max_queue_size=2*batch_size,
                                  workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

    return model, history


if __name__ == '__main__':
    parser = ArgumentParser(description="Train a 3D Model unet")
    parser.add_argument("-l", "--logdir", help="Directory where to log the data for tensorboard",
                        default="/home/simon/")

    parser.add_argument("-d", "--data_path", help="Path to data folder",
                        default="/home/simon/Datasets/HUG/32x32x32")
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument("-s", "--skip_blank", type=bool, help="Skip blank images - will not be fed to the network", default=False)
    parser.add_argument("-e", "--epoch_size", type=int, help="Steps per epoch", default=None)
    parser.add_argument("-p", "--patient", type=int, help="Patient from which to log an image in tensorboard", default=295742)

    args = parser.parse_args()
    data_path = args.data_path
    logdir = os.path.join(args.logdir, time.strftime("%Y%m%d_%H-%M-%S", time.gmtime()))
    batch_size = args.batch_size
    num_patient = args.patient

    # Get patch size
    path_train = os.path.join(data_path, "train")
    input_example = os.listdir(os.path.join(path_train, "MTT"))[0]
    patch_size = nb.load(os.path.join(path_train, "MTT", input_example)).get_data().shape

    if(args.skip_blank):
        print("Skipping blank images")
    print("Patch size detected : {}".format(patch_size))

    metrics = [
               weighted_dice_coefficient,
               dice_coefficient,
               tversky_coeff,
               'acc',
               'mse',
               ]

    loss_function = tversky_loss

    folders_input = ["CBV", "CBF", "MTT", "Tmax"]
    #folders_input = ["Tmax"]
    folders_target = ["lesion"]

    model = unet_model_3d([len(folders_input), patch_size[0], patch_size[1], patch_size[2]],
                          pool_size=[2, 2, 2],
                          n_base_filters=16,
                          depth=3,
                          batch_normalization=False,
                          metrics=metrics,
                          loss=loss_function,
                          activation_name="sigmoid")

    create_if_not_exists(logdir)
    train(model, batch_size=batch_size, data_path=data_path, logdir=logdir,
          skip_blank=args.skip_blank, epoch_size=args.epoch_size, patch_size=patch_size,
          folders_input=folders_input, folders_target=folders_target, num_patient=num_patient)
