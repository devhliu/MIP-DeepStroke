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
import keras
from metrics import dice_coefficient, weighted_dice_coefficient, weighted_dice_coefficient_loss, dice_coefficient_loss, tversky_coeff, tversky_loss
from keras.metrics import binary_crossentropy, binary_accuracy, mean_absolute_error
import tensorflow as tf
import json
from keras.callbacks import ReduceLROnPlateau
from image_augmentation import randomly_augment

config = tf.ConfigProto(device_count={'GPU': 1 , 'CPU': 1} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)


def create_generators(batch_size, data_path=None, skip_blank=True, folders_input=['input'], folders_target=['lesion'], augment_prob=0.0):
    train_path = "train/"
    validation_path = "validation/"
    if data_path is not None:
        train_path = os.path.join(data_path, train_path)
        validation_path = os.path.join(data_path, validation_path)

    training_size = len(os.listdir(os.path.join(train_path, "MTT")))
    validation_size = len(os.listdir(os.path.join(validation_path, "MTT")))

    print("Train data path {} - {} samples".format(train_path, training_size))
    print("Validation data path {} - {} samples".format(validation_path, validation_size))

    train_generator = dual_generator(train_path, folders_input, folders_target, batch_size=batch_size, skip_blank=skip_blank, augment_prob=augment_prob)

    validation_generator = dual_generator(validation_path, folders_input, folders_target, batch_size=batch_size, skip_blank=skip_blank)

    return train_generator, validation_generator


def dual_generator(data_directory, folders_input, folders_target, batch_size, skip_blank=False, logfile=None, augment_prob=0.0):
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

            # perform data augmentation
            inputs, targets = randomly_augment(inputs, targets, prob=augment_prob)

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


def train(model, data_path, batch_size=32, logdir=None, skip_blank=True, epoch_size=None, patch_size=None, folders_input=['input'], folders_target=['lesion'],
          test_patient=295742, train_patient=758594, learning_rate_patience=20, learning_rate_decay=0.0, stage="wcoreg_",
          augment_prob=0.0):

    training_generator, validation_generator = create_generators(batch_size, data_path=data_path, skip_blank=skip_blank,
                                                                 folders_input=folders_input,
                                                                 folders_target=folders_target, augment_prob=augment_prob)

    dataset_training_size = len(os.listdir(os.path.join(data_path, "train",folders_input[0])))
    dataset_val_size = len(os.listdir(os.path.join(data_path, "validation",folders_input[0])))

    tensorboard_callback = None
    if logdir is not None:
        log_path = create_if_not_exists(os.path.join(logdir, "logs"))

        # Define patient paths and layers to display
        patient_path1 = "/home/snarduzz/Data/preprocessed_original_masked/{}".format(train_patient)
        patient_path2 = "/home/snarduzz/Data/preprocessed_original_masked/{}".format(test_patient)

        patients = dict({
            patient_path1: ["train", [35]],
            patient_path2: ["validation", [35]]
        })

        training_generator_log, validation_generator_log = create_generators(batch_size, data_path=data_path,
                                                                     skip_blank=skip_blank,
                                                                     folders_input=folders_input,
                                                                    folders_target=folders_target, augment_prob=augment_prob)

        tensorboard_callback = TrainValTensorBoard(log_dir=log_path,
                                                   training_generator=training_generator_log,
                                                   validation_generator=validation_generator_log,
                                                   validation_steps= int(dataset_val_size/batch_size),
                                                   patients=patients,
                                                   patch_size=patch_size,
                                                   folders_input=folders_input,
                                                   folders_target=folders_target,
                                                   verbose=1,
                                                   histogram_freq=1,
                                                   batch_size=batch_size,
                                                   write_graph=True,
                                                   write_grads=True,
                                                   write_images=True,
                                                   embeddings_freq=0,
                                                   embeddings_layer_names=None,
                                                   embeddings_metadata=None,
                                                   stage=stage)


        # Start Tensorboard
        print("\033[94m" + "tensorboard --logdir={}".format(log_path) + "\033[0m")

    # Save checkpoint each 5 epochs
    checkpoint_path = create_if_not_exists(os.path.join(logdir, "checkpoints"))
    checkpoint_filename = os.path.join(checkpoint_path, "model.{epoch:02d}-{val_loss:.2f}.hdf5")

    checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_filename, monitor='val_loss', verbose=0, save_best_only=False,
                                    save_weights_only=False, mode='auto', period=1)

    LRReduce = ReduceLROnPlateau(factor=learning_rate_decay, patience=learning_rate_patience)

    # Parameters
    validation_steps = 1   # Number of steps per evaluation (number of to pass)
    steps_per_epoch = (dataset_training_size/batch_size)  # Number of batches to pass before going to next epoch
    if epoch_size is not None :
        steps_per_epoch = epoch_size

    # Train the model, iterating on the data in batches of 32 samples
    with tf.device("/device:GPU:{}".format(GPU_ID)):
        history = model.fit_generator(training_generator, steps_per_epoch=steps_per_epoch, epochs=1000, verbose=1,
                                      callbacks=[tensorboard_callback, checkpoint_callback, LRReduce],
                                      validation_data=validation_generator, validation_steps=validation_steps,
                                      class_weight=None, max_queue_size=2*batch_size,
                                      workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

    return model, history


if __name__ == '__main__':
    parser = ArgumentParser(description="Train a 3D Model unet")
    parser.add_argument("-l", "--logdir", help="Directory where to log the data for tensorboard",
                        default="/home/snarduzz/Models")

    parser.add_argument("-d", "--data_path", help="Path to data folder",
                        default="/home/snarduzz/Data")
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument("-s", "--skip_blank", type=bool, help="Skip blank images - will not be fed to the network", default=False)
    parser.add_argument("-e", "--epoch_size", type=int, help="Steps per epoch", default=None)
    parser.add_argument("-testp", "--test_patient", type=int, help="Patient from which to log an image in tensorboard", default=295742)
    parser.add_argument("-trainp", "--train_patient", type=int, help="Patient from which to log an image in tensorboard", default=758594)
    parser.add_argument("-lr", "--initial_learning_rate", type=float, help="Initial learning rate", default=1e-6)
    parser.add_argument("-a", "--activation_name", type=str, help="activation name", default="sigmoid")
    parser.add_argument("-la", "--layer_activation", type=str, help="layer activation name", default="relu")
    parser.add_argument("-f", "--filters", type=int, help="number of base filters", default=16)
    parser.add_argument("-gpu", "--gpu", type=int, help="GPU number", default=0)
    parser.add_argument("-decay", "--decay", type=float, help="Decay rate of learning", default=0.0)
    parser.add_argument("-depth", "--depth", type=float, help="Depth of Unet", default=5)
    parser.add_argument("-bn", "--batch_normalization", type=bool, help="Activate batch normalization", default=False)
    parser.add_argument("-loss", "--loss", type=str, help="Loss function : [tversky, dice, weighted_dice, mean_absolute_error]", default="tversky")
    parser.add_argument("-params", "--parameters", type=str, help="path to JSON containing the parameters of the model", default=None)
    parser.add_argument('-i', '--input', nargs='+', action="append", help='Input : use -i T2, -i Tmax, -i CBV -i CBF, -i MTT', required=True)
    parser.add_argument('-o', '--output', nargs='+', action="append", help='Input : use -o lesion', required=True)
    parser.add_argument('-stage','--stage', help="Stage of registration : nothing, coreg_ or wcoreg_", default="wcoreg_")
    parser.add_argument('-augment', '--augment', help="Augmentation probability", default=0.0)
    parser.add_argument('-patience', '--learning_rate_patience', help="Learning rate patience", type=int, default=10)

    args = parser.parse_args()
    logdir = os.path.join(args.logdir, time.strftime("%Y%m%d_%H-%M-%S", time.gmtime()))
    create_if_not_exists(logdir)

    # If parameters are not specified, load from command line arguments
    if args.parameters is None:
        parameters = dict()
        parameters["logdir"] = logdir
        parameters["data_path"] = args.data_path
        parameters["batch_size"] = args.batch_size
        parameters["batch_normalization"] = args.batch_normalization
        parameters["skip_blank"] = args.skip_blank
        parameters["steps_per_epoch"] = args.epoch_size
        parameters["initial_learning_rate"] = args.initial_learning_rate
        parameters["decay"] = args.decay
        parameters["final_activation"] = args.activation_name
        parameters["n_filters"] = args.filters
        parameters["depth"] = args.depth
        parameters["GPU_ID"] = args.gpu
        parameters["loss_function"] = args.loss
        parameters["test_patient"] = args.test_patient
        parameters["train_patient"] = args.train_patient
        parameters["inputs"] = [x[0] for x in args.input]
        parameters["targets"] = [x[0] for x in args.output]
        parameters["stage"] = args.stage
        parameters["augment_prob"] = args.augment
        parameters["layer_activation"] = args.layer_activation
    else:
        # If parameters are specified, load them from JSON
        print("Loading parameters from : "+args.parameters)
        with open(args.parameters, 'r') as fp:
            parameters = json.load(fp)

    # Save copy of json in folder of the model
    json_file = os.path.join(logdir, "parameters.json")
    print("Saving parameters in : "+json_file)
    with open(json_file, 'w') as fp:
        json.dump(parameters, fp)

    #Load values from parameters
    data_path = parameters["data_path"]
    batch_size = parameters["batch_size"]
    batch_normalization = parameters["batch_normalization"]
    skip_blank = parameters["skip_blank"]
    steps_per_epoch = parameters["steps_per_epoch"]
    initial_learning_rate = parameters["initial_learning_rate"]
    decay = parameters["decay"]
    final_activation = parameters["final_activation"]
    n_filters = parameters["n_filters"]
    depth = parameters["depth"]
    GPU_ID = parameters["GPU_ID"]
    loss_function = parameters["loss_function"]
    test_patient = parameters["test_patient"]
    train_patient = parameters["train_patient"]
    inputs = [x[0] for x in args.input]
    targets = [x[0] for x in args.output]
    if "stage" in parameters.keys():
        stage = parameters["stage"]
    else:
        stage = args.stage
    if "augment_prob" in parameters.keys():
        augment_prob = parameters["augment_prob"]
    else:
        augment_prob = args.augment
    if "layer_activation" in parameters.keys():
        layer_activation = parameters["layer_activation"]
    else:
        layer_activation = args.layer_activation
    if "learning_rate_patience" in parameters.keys():
        learning_rate_patience = parameters["learning_rate_patience"]
    else:
        learning_rate_patience = args.learning_rate_patience

    #Display Parameters
    print("---")
    print("Parameters")
    for key in parameters.keys():
        print("{} : {}".format(key,parameters[key]))
    print("---")

    if len(inputs) == 0 or len(targets) == 0:
        raise Exception("Please provide inputs and outputs channels")

    print('\033[1m' + "INPUTS : " + str(inputs) + '\033[0m')
    print('\033[1m' + "TARGETS : " + str(targets) + '\033[0m')

    # Set the script to use GPU with GPU_ID
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    # Get patch size
    path_train = os.path.join(data_path, "train")
    input_example = os.listdir(os.path.join(path_train, "MTT"))[0]
    patch_size = nb.load(os.path.join(path_train, "MTT", input_example)).get_data().shape

    if(skip_blank):
        print("Skipping blank images")
    print("Patch size detected : {}".format(patch_size))

    metrics = [
               weighted_dice_coefficient,
               dice_coefficient,
               tversky_coeff,
               'acc',
               'mse',
                binary_crossentropy,
                binary_accuracy
               ]

    losses = {
        "tversky":tversky_loss,
        "dice":dice_coefficient_loss,
        "weighted_dice":weighted_dice_coefficient_loss,
        "mean_absolute_error" : mean_absolute_error
    }

    loss_function = losses[loss_function]

    model = unet_model_3d([len(inputs), patch_size[0], patch_size[1], patch_size[2]],
                          pool_size=[2, 2, 2],
                          n_base_filters=n_filters,
                          depth=depth,
                          batch_normalization=batch_normalization,
                          metrics=metrics,
                          initial_learning_rate=initial_learning_rate,
                          loss=loss_function,
                          final_activation_name=final_activation,
                          layer_activation_name=layer_activation)

    train(model, batch_size=batch_size, data_path=data_path, logdir=logdir,
          skip_blank=skip_blank, epoch_size=steps_per_epoch, patch_size=patch_size,
          folders_input=inputs, folders_target=targets, test_patient=test_patient,
          train_patient=train_patient, learning_rate_patience=learning_rate_patience, learning_rate_decay=decay,
          stage=stage, augment_prob=augment_prob)
