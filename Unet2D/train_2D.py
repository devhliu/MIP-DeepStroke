import os
import nibabel as nb
import numpy as np
from utils2D import create_if_not_exists, transorm_to_2d_usable_data
from callbacks import TrainValTensorBoard
from argparse import ArgumentParser
import time
import keras
from metrics import dice_coefficient, weighted_dice_coefficient, weighted_dice_coefficient_loss, dice_coefficient_loss, \
    tversky_coeff, tversky_loss
from keras.metrics import binary_crossentropy, binary_accuracy, mean_absolute_error
import tensorflow as tf
import json
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import SGD
import losses as model_losses
from image_augmentation import randomly_augment
import newmodels
from keras.preprocessing.image import ImageDataGenerator

from data_generator import DataGenerator

config = tf.ConfigProto(device_count={'GPU': 2, 'CPU': 1})
sess = tf.Session(config=config)
keras.backend.set_session(sess)


def create_generators(batch_size, data_path=None, skip_blank=True, folders_input=['input'], folders_target=['lesion'],
                      augment_prob=None, depth=4):
    train_path = "train/"
    validation_path = "validation/"
    if data_path is not None:
        train_path = os.path.join(data_path, train_path)
        validation_path = os.path.join(data_path, validation_path)

    training_size = len(os.listdir(os.path.join(train_path, folders_input[0])))
    validation_size = len(os.listdir(os.path.join(validation_path, folders_input[0])))

    print("Train data path {} - {} samples".format(train_path, training_size))
    print("Validation data path {} - {} samples".format(validation_path, validation_size))

    train_generator = DataGenerator(data_directory=train_path,
                                     folders_input=folders_input,
                                     folders_output=folders_target,
                                     batch_size=batch_size,
                                     augment_prob=augment_prob, depth=depth)

    validation_generator = DataGenerator(data_directory=validation_path,
                                          folders_input=folders_input,
                                          folders_output=folders_target,
                                          batch_size=batch_size,
                                          augment_prob=augment_prob, depth=depth)

    return train_generator, validation_generator


def dual_generator(data_directory, folders_input, folders_output, batch_size, skip_blank=False, logfile=None,
                   augment_prob=None, depth=4):
    # default values
    if augment_prob is None:
        augment_prob = {"rotation": 0.0,
                        "rotxmax": 90.0,
                        "rotymax": 90.0,
                        "rotzmax": 90.0,
                        "rotation_step": 1.0,
                        "salt_and_pepper": 0.0,
                        "flip": 0.0,
                        "contrast_and_brightness": 0.0,
                        "only_positives": False}

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
            for y in folders_output:
                target_path = image_paths[i].replace(folders_input[0], y)
                image_target = nb.load(os.path.join(data_directory, y, target_path)).get_data()
                targets.append(image_target)
                targets_paths.append(target_path)
            if logfile:
                with open(logfile, "a") as f:
                    paths = inputs_paths + targets_paths
                    f.write("{} - {}".format(i, paths))

            # perform data augmentation
            if augment_prob["only_positives"] and np.sum(targets) > 0:
                inputs, targets = randomly_augment(inputs, targets, prob=augment_prob)
            else:
                inputs, targets = randomly_augment(inputs, targets, prob=augment_prob)

            if not (np.all(targets == 0) and skip_blank):
                x_list.append(inputs)
                y_list.append(targets)

            if len(x_list) == batch_size:
                x_list_batch = np.array(x_list)
                y_list_batch = np.array(y_list)
                x_list = []
                y_list = []

                yield transorm_to_2d_usable_data(x_list_batch, y_list_batch, depth=depth)

        while len(x_list) < batch_size:
            input_size = nb.load(os.path.join(data_directory, folders_input[0], image_paths[0])).get_data().shape
            zero = np.zeros(input_size)

            inputs = []
            for x in folders_input:
                inputs.append(zero)
            targets = []
            for y in folders_output:
                targets.append(zero)

            x_list.append(inputs)
            y_list.append(targets)
        yield transorm_to_2d_usable_data(np.array(x_list), np.array(y_list), depth=depth)


def train(model, data_path, batch_size=32, logdir=None, skip_blank=True, epoch_size=None, max_epochs= 1000, patch_size=None,
          folders_input=['input'], folders_target=['lesion'],
          test_patient="/home/snarduzz/Data/Data_2016_T2_TRACE_LESION/97623138",
          train_patient="/home/snarduzz/Data/Data_2016_T2_TRACE_LESION/898729",
          learning_rate_patience=20, learning_rate_decay=0.0, stage="rcoreg_",
          augment_prob=None,
          depth=4):
    if augment_prob is None:
        augment_prob = {"rotation": args.augment,
                        "rotxmax": 90.0,
                        "rotymax": 90.0,
                        "rotzmax": 90.0,
                        "rotation_step": 1.0,
                        "salt_and_pepper": args.augment,
                        "flip": args.augment,
                        "contrast_and_brightness": args.augment,
                        "only_positives": True}

    training_generator, validation_generator = create_generators(batch_size, data_path=data_path, skip_blank=skip_blank,
                                                                 folders_input=folders_input,
                                                                 folders_target=folders_target,
                                                                 augment_prob=augment_prob, depth=depth)

    dataset_training_size = len(os.listdir(os.path.join(data_path, "train", folders_input[0])))
    dataset_val_size = len(os.listdir(os.path.join(data_path, "validation", folders_input[0])))

    tensorboard_callback = None
    if logdir is not None:
        log_path = create_if_not_exists(os.path.join(logdir, "logs"))

        # Define patient paths and layers to display
        patient_path1 = train_patient
        patient_path2 = test_patient
        p1 = os.path.basename(train_patient)
        p2 = os.path.basename(test_patient)
        if not p1.isdigit() or not p2.isdigit():
            raise Exception("Patients for validation should be digits")

        img_test_path = os.path.join(patient_path1, "Neuro_Cerebrale_64Ch/{}VOI_lesion_{}.nii".format(stage, p1))
        patient_img = nb.load(img_test_path).get_data()

        layer = int(patient_img.shape[2] / 2)
        patients = dict({
            patient_path1: ["train", [layer]],
            patient_path2: ["validation", [layer]]
        })

        training_generator_log, validation_generator_log = create_generators(batch_size, data_path=data_path,
                                                                             skip_blank=skip_blank,
                                                                             folders_input=folders_input,
                                                                             folders_target=folders_target,
                                                                             augment_prob=augment_prob)

        tensorboard_callback = TrainValTensorBoard(log_dir=log_path,
                                                   training_generator=training_generator_log,
                                                   validation_generator=validation_generator_log,
                                                   validation_steps=int(dataset_val_size / batch_size),
                                                   patients=patients,
                                                   patch_size=patch_size,
                                                   folders_input=folders_input,
                                                   folders_target=folders_target,
                                                   verbose=1,
                                                   histogram_freq=0,
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
    checkpoint_filename = os.path.join(checkpoint_path, "model.{epoch:02d}-{val_dsc:.4f}-dsc.hdf5")

    checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_filename, monitor='val_dsc', verbose=0,
                                                          save_best_only=False,
                                                          save_weights_only=False, mode='auto', period=1)

    LRReduce = ReduceLROnPlateau(factor=learning_rate_decay, patience=learning_rate_patience)

    # Parameters
    validation_steps = (dataset_val_size / batch_size)  # Number of steps per evaluation (number of to pass)
    steps_per_epoch = (dataset_training_size / batch_size)  # Number of batches to pass before going to next epoch
    if epoch_size is not None:
        steps_per_epoch = epoch_size

    # Train the model, iterating on the data in batches of 32 samples
    history = model.fit_generator(training_generator, steps_per_epoch=steps_per_epoch, epochs=max_epochs, verbose=1,
                                  callbacks=[tensorboard_callback, checkpoint_callback, LRReduce],
                                  validation_data=validation_generator, validation_steps=validation_steps,
                                  class_weight=None, max_queue_size=2 * batch_size,
                                  workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

    return model, history


if __name__ == '__main__':
    parser = ArgumentParser(description="Train a 3D Model unet")
    parser.add_argument("-l", "--logdir", help="Directory where to log the data for tensorboard",
                        default="/home/snarduzz/Models")

    parser.add_argument("-d", "--data_path", help="Path to data folder",
                        default="/home/snarduzz/Data")
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument("-s", "--skip_blank", type=bool, help="Skip blank images - will not be fed to the network",
                        default=False)
    parser.add_argument("-e", "--epoch_size", type=int, help="Steps per epoch", default=None)
    parser.add_argument("-testp", "--test_patient", type=str, help="Patient from which to log an image in tensorboard",
                        default="/home/snarduzz/Data/Data_For_2D/97623138")
    parser.add_argument("-trainp", "--train_patient", type=str,
                        help="Patient from which to log an image in tensorboard",
                        default="/home/snarduzz/Data/Data_For_2D/898729")
    parser.add_argument("-lr", "--initial_learning_rate", type=float, help="Initial learning rate", default=1e-6)
    parser.add_argument("-a", "--activation_name", type=str, help="activation name", default="sigmoid")
    parser.add_argument("-la", "--layer_activation", type=str, help="layer activation name", default="relu")
    parser.add_argument("-f", "--filters", type=int, help="number of base filters", default=16)
    parser.add_argument("-gpu", "--gpu", type=int, help="GPU number", default=0)
    parser.add_argument("-decay", "--decay", type=float, help="Decay rate of learning", default=0.0)
    parser.add_argument("-depth", "--depth", type=float, help="Depth of Unet", default=5)
    parser.add_argument("-bn", "--batch_normalization", type=bool, help="Activate batch normalization", default=False)
    parser.add_argument("-loss", "--loss", type=str,
                        help="Loss function : [tversky, dice, weighted_dice, mean_absolute_error]", default="tversky")
    parser.add_argument("-params", "--parameters", type=str, help="path to JSON containing the parameters of the model",
                        default=None)
    parser.add_argument('-i', '--input', nargs='+', action="append",
                        help='Input : use -i T2, -i Tmax, -i CBV -i CBF, -i MTT', required=True)
    parser.add_argument('-o', '--output', nargs='+', action="append", help='Input : use -o lesion', required=True)
    parser.add_argument('-stage', '--stage', help="Stage of registration : nothing, coreg_ or wcoreg_",
                        default="rcoreg_")
    parser.add_argument('-augment', '--augment', help="Augmentation probability", default=0.0)
    parser.add_argument('-patience', '--learning_rate_patience', help="Learning rate patience", type=int, default=10)
    parser.add_argument('-architecture', '--architecture', help="Model : 3dunet or isensee17", default="3dunet")

    args = parser.parse_args()
    logdir = os.path.join(args.logdir, time.strftime("%Y%m%d_%H-%M-%S", time.gmtime()))
    create_if_not_exists(logdir)

    # If parameters are not specified, load from command line arguments
    if args.parameters is None:
        parameters = dict()
        parameters["max_epochs"] = 1000
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
        parameters["augment_prob"] = {"rotation": float(args.augment),
                                      "rotxmax": 180.0,
                                      "rotation_step": 15.0,
                                      "salt_and_pepper": float(args.augment),
                                      "flip": float(args.augment),
                                      "zoom": float(args.augment),
                                      "contrast_and_brightness": float(args.augment),
                                      "only_positives": False}
        parameters["dropout"] = 0.0

        parameters["layer_activation"] = args.layer_activation
        parameters["architecture"] = args.architecture
        if parameters["loss_function"] == "tversky":
            parameters["tversky_alpha-beta"] = (0.5, 0.5)
    else:
        # If parameters are specified, load them from JSON
        print("Loading parameters from : " + args.parameters)
        with open(args.parameters, 'r') as fp:
            parameters = json.load(fp)


    alpha_value, beta_value = parameters["tversky_alpha-beta"]
    print("alpha = {}, beta = {}".format(alpha_value,beta_value))
    # Load values from parameters
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
    architecture = parameters["architecture"]
    dropout = parameters["dropout"]
    max_epochs = parameters["max_epochs"]

    inputs = [x[0] for x in args.input]
    targets = [x[0] for x in args.output]
    if "stage" in parameters.keys():
        stage = parameters["stage"]
    else:
        stage = args.stage
    if "augment_prob" in parameters.keys():
        augment_prob = parameters["augment_prob"]
    else:
        augment_prob = {"rotation": args.augment,
                        "rotxmax": 90.0,
                        "rotation_step": 30.0,
                        "salt_and_pepper": args.augment,
                        "flip": args.augment,
                        "zoom": 1.0,
                        "contrast_and_brightness": args.augment,
                        "only_positives": False}
        parameters["augment_prob"] = augment_prob

    if "layer_activation" in parameters.keys():
        layer_activation = parameters["layer_activation"]
    else:
        layer_activation = args.layer_activation
        parameters["layer_activation"] = layer_activation

    if "learning_rate_patience" in parameters.keys():
        learning_rate_patience = parameters["learning_rate_patience"]
    else:
        learning_rate_patience = args.learning_rate_patience
        parameters["learning_rate_patience"] = learning_rate_patience

    # Save copy of json in folder of the model
    json_file = os.path.join(logdir, "parameters.json")
    print("Saving parameters in : " + json_file)
    with open(json_file, 'w') as fp:
        json.dump(parameters, fp, indent=4)

    # Display Parameters
    print("---")
    print("Parameters")
    for key in parameters.keys():
        print("{} : {}".format(key, parameters[key]))
    print("---")

    if len(inputs) == 0 or len(targets) == 0:
        raise Exception("Please provide inputs and outputs channels")

    print('\033[1m' + "INPUTS : " + str(inputs) + '\033[0m')
    print('\033[1m' + "TARGETS : " + str(targets) + '\033[0m')

    if GPU_ID > 0:
        # Set the script to use GPU with GPU_ID
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    # Get patch size
    path_train = os.path.join(data_path, "train")
    input_example = os.listdir(os.path.join(path_train, inputs[0]))[0]
    patch_size = nb.load(os.path.join(path_train, inputs[0], input_example)).get_data().shape

    if (skip_blank):
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

    if loss_function == "tversky":
        loss_function = model_losses.get_tversky(alpha_value, beta_value)
    elif loss_function == "dice":
        loss_function = model_losses.dice_loss
    elif loss_function =="jaccard":
        loss_function = model_losses.jaccard_distance
    else:
        print("Loading default : dice loss")
        loss_function = model_losses.dice_loss
        parameters["loss_function"] = "dice (default)"
        # Save copy of json in folder of the model
        json_file = os.path.join(logdir, "parameters.json")
        print("Saving parameters in : " + json_file)
        with open(json_file, 'w') as fp:
            json.dump(parameters, fp, indent=4)


    sgd = SGD(lr=initial_learning_rate, momentum=0.9)

    # load model : default is 3dunet
    if architecture == "attn_reg":
        model = newmodels.attn_reg(sgd, input_size=[patch_size[0], patch_size[1], len(inputs)], lossfxn=loss_function)
    elif architecture == "attn_reg_ds":
        model = newmodels.attn_reg_ds(sgd, input_size=[patch_size[0], patch_size[1], len(inputs)], lossfxn=loss_function)
    elif architecture == "attn_unet":
        model = newmodels.attn_unet(sgd, input_size=[patch_size[0], patch_size[1], len(inputs)], lossfxn=loss_function)
    else:
        print("Loading default : Unet")
        parameters["architecture"] = "Unet (default)"
        model = newmodels.unet(sgd, input_size=[patch_size[0], patch_size[1], len(inputs)],
                                   lossfxn=loss_function)
        # Save copy of json in folder of the model
        json_file = os.path.join(logdir, "parameters.json")
        print("Saving parameters in : " + json_file)
        with open(json_file, 'w') as fp:
            json.dump(parameters, fp, indent=4)

    dimensions = len(np.array(model.output_shape).shape)
    if dimensions>1:
        model_output_depth = len(model.output_shape)
    else:
        model_output_depth = 1
    print("Model output depth ",model_output_depth)

    train(model, batch_size=batch_size, data_path=data_path, logdir=logdir,
          skip_blank=skip_blank, epoch_size=steps_per_epoch, max_epochs=max_epochs, patch_size=patch_size,
          folders_input=inputs, folders_target=targets, test_patient=test_patient,
          train_patient=train_patient, learning_rate_patience=learning_rate_patience, learning_rate_decay=decay,
          stage=stage, augment_prob=augment_prob, depth=model_output_depth)
