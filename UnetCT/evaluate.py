import numpy as np
import nibabel as nb
import sys
sys.path.append('../')
from UnetCT.predict import predict_patch
import os
from argparse import ArgumentParser
from tqdm import tqdm
from keras.models import load_model
from UnetCT.metrics import *
import sklearn.metrics as metrics
from datetime import datetime
import pandas as pd
import json
import re
import traceback
import tensorflow as tf

def parseLoss(model_name):
    split = model_name.split("-")
    if len(split)>2:
        score = split[2].replace(".hdf5","")
    else:
        score = split[1].replace(".hdf5", "")
    score = float(score)
    if score < 0:
        score = -score
    return score

def parseIteration(model_name):
    iteration = model_name.split("-")[0].replace("model.","")
    return iteration


def specificity(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def dice_score(y_true,y_pred):
    dice = np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))
    return dice

def tversky_score(y_true, y_pred, alpha=0.3):
    onlyA = np.sum(y_pred)
    onlyB = np.sum(y_true)
    bothAB = np.sum(y_pred[y_true == 1])

    tversky = bothAB / (alpha*onlyA + (1-alpha)*onlyB + bothAB)
    return tversky

def predict_patient(patient_id, list_files, model, channels_input=["T2"], channels_output=["lesion"]):
    patient_patches_names = [x for x in list_files if patient_id in x]
    patch_size = nb.load(patient_patches_names[0]).get_data().shape

    ys_true = []
    ys_pred = []

    input_option = channels_input[0]
    for i in tqdm(range(len(patient_patches_names))):

        inputs_patches = np.empty(shape=[len(channels_input), patch_size[0], patch_size[1], patch_size[2]])
        for index_c in range(len(channels_input)):
            c = channels_input[index_c]
            x_file = patient_patches_names[i].replace(input_option,c)
            x = nb.load(x_file).get_data()
            inputs_patches[index_c, :, :, :] = x

        output_patches = np.empty(shape=[len(channels_output), patch_size[0], patch_size[1], patch_size[2]])
        for index_c in range(len(channels_output)):
            c = channels_output[index_c]
            y_file = patient_patches_names[i].replace(input_option, c)
            y = nb.load(y_file).get_data().astype(np.int8)
            output_patches[index_c, :, :, :] = y

        # Predict one patch
        y_pred = predict_patch(inputs_patches, model=model)

        ys_true += list(output_patches.flatten())
        ys_pred += list(y_pred.flatten())

    return np.array(ys_true), np.array(ys_pred)


def predict(test_folder, model, maxsize=None, channels_input=["T2"], channels_output=["lesion"]):
    input_option = channels_input[0]
    input_files = [os.path.join(test_folder, input_option, x) for x in os.listdir(os.path.join(test_folder, input_option))]

    s = nb.load(input_files[0]).get_data().size

    if maxsize is None:
        maxsize = len(input_files)

    patient_list = list(set([re.search('%s(.*)%s' % ("{}_".format(input_option), "-"), x).group(1) for x in input_files]))

    files_per_patients = len([x for x in input_files if patient_list[0] in x])

    list_y = np.empty(len(patient_list)*files_per_patients*s)
    list_y_pred = np.empty(len(patient_list)*files_per_patients*s)

    aucs = []
    f1_scores = []
    dices = []
    tverskys = []

    for i in tqdm(range(len(patient_list))):
        p = patient_list[i]
        y_true, y_pred = predict_patient(p, input_files, model, channels_input, channels_output)

        auc = metrics.roc_auc_score(y_true, y_pred)

        # Treshold
        y_pred_thresh = y_pred.copy()
        y_pred_thresh[y_pred_thresh < 0.5] = 0
        y_pred_thresh[y_pred_thresh >= 0.5] = 1
        try:
            f1 = metrics.f1_score(y_true, y_pred_thresh)
        except:
            f1 = 0.0
            print("F1 is ill-posed : set to 0.0")

        # Convert to Tensor for Dice and Tversky scores
        y_true_tensor = tf.cast(y_true, tf.float64)
        y_pred_tensor = tf.cast(y_pred, tf.float64)
        dice = dice_score(y_true=y_true, y_pred=y_pred)
        tversky = tversky_score(y_true=y_true, y_pred=y_pred)

        aucs.append(auc)
        f1_scores.append(f1)
        dices.append(dice)
        tverskys.append(tversky)

        list_y[i*len(y_true):(i+1)*len(y_true)] = y_true
        list_y_pred[i * len(y_pred):(i + 1) * len(y_pred)] = y_pred

    y_true = list_y
    y_pred = list_y_pred

    # Normal scoring on whole test set without threshold
    auc = metrics.roc_auc_score(y_true,  y_pred)
    coeff_dice = dice_score(y_true, y_pred)
    coeff_tversky = tversky_score(y_true, y_pred)

    dict_scores = {"auc": auc}
    dict_scores["dice"] = coeff_dice
    dict_scores["tversky"] = coeff_tversky

    # AUC per patient with std
    dict_scores["auc_mean"] = np.mean(np.array(aucs))
    dict_scores["auc_std"] = np.std(np.array(aucs))
    dict_scores["f1_mean"] = np.mean(np.array(f1_scores))
    dict_scores["f1_std"] = np.std(np.array(f1_scores))
    dict_scores["dice_mean"] = np.mean(np.array(dices))
    dict_scores["dice_std"] = np.std(np.array(dices))
    dict_scores["tversky_mean"] = np.mean(np.array(tverskys))
    dict_scores["tversky_std"] = np.std(np.array(tverskys))

    functions = {"dice_thresh": dice_score,
                 "tversky_thresh": tversky_score,
                 "ap":metrics.average_precision_score,
                 "f1-score":metrics.f1_score,
                 "jaccard":metrics.jaccard_similarity_score,
                 "accuracy":metrics.accuracy_score,
                 "precision":metrics.precision_score,
                 "sensitivity":metrics.recall_score,
                 "specificity":specificity
                }

    # Tresholding
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1

    for k in tqdm(functions.keys()):
        if k is "f1-score":
            try:
                score = functions[k](y_true, y_pred)
            except:
                score = "NaN"
        else:
            try:
                score = functions[k](y_true, y_pred)
            except Exception as e:
                print("error computing {}".format(k))
                score = "NaN"
        dict_scores[k] = score

    return dict_scores


def evaluate_dir(logdir, channels_input, channels_output, to_replace={"/home/snarduzz/Data":"/home/snarduzz/Data"}):
    checkpoints_folder = os.path.join(logdir, "checkpoints")
    parameters_file = os.path.join(logdir, "parameters.json")
    output_file = os.path.join(logdir, "evaluation.csv")
    date = os.path.basename(logdir)
    # load parameters
    print("Loading parameters from : " + parameters_file)
    with open(parameters_file, 'r') as fp:
        parameters = json.load(fp)

    path_replaced = parameters["data_path"]
    if not os.path.exists(path_replaced):
        # try by replacing the value
        for k in to_replace.keys():
            path_replaced = path_replaced.replace(k, to_replace[k])

    # if still not valid
    if os.path.exists(path_replaced):
        raise Exception("Path to data {} not found.".format(path_replaced))

    data_path = os.path.join(path_replaced, "test")

    # Create DF if not exists
    if not os.path.exists(output_file):
        df = pd.DataFrame(columns=columns).reset_index()
        df.to_csv(output_file)

        # evaluate each checkpoints
        checkpoints = [os.path.join(checkpoints_folder, x) for x in os.listdir(checkpoints_folder)]
        if (len(checkpoints) < 1):
            print("No checkpoints found in {}".format(checkpoints_folder))
        else:
            print("{} checkpoints found in folder.".format(len(checkpoints)))
        # For all checkpoints, evaluate.
        for ckpt in tqdm(checkpoints):

            # put extra information
            model_name = os.path.basename(ckpt)
            iteration = parseIteration(model_name)
            loss_value = parseLoss(model_name)

            # Load CSV and append line
            df = pd.read_csv(output_file, header=0)

            # Check that the model was not already tested
            if model_name in df["model_name"].values and date in df["date"].values:
                print(date, model_name, "already tested.")
                continue

            model = load_model(ckpt, custom_objects={"dice_coefficient": dice_coefficient,
                                                     "dice_coefficient_loss": dice_coefficient_loss,
                                                     "weighted_dice_coefficient_loss": weighted_dice_coefficient_loss,
                                                     "weighted_dice_coefficient": weighted_dice_coefficient,
                                                     "tversky_loss": tversky_loss,
                                                     "tversky_coeff": tversky_coeff
                                                     })
            dict_scores = {}
            try:
                dict_scores = predict(data_path, model, channels_input, channels_output)
            except Exception as e:
                print("Error while predicting model {}. Try with another image patch size.".format(ckpt))
                traceback.print_exc()
                continue

            dict_scores["model_name"] = model_name
            dict_scores["date"] = date
            dict_scores["iteration"] = iteration
            dict_scores["val_loss"] = loss_value
            dict_scores["loss_function"] = parameters["loss_function"]
            dict_scores["parameters_file"] = parameters_file

            # Append line to CSV by keeping only relevant columns
            df = df[list(dict_scores.keys())]
            new_df = pd.DataFrame(dict_scores, index=[0])
            df = df.append(new_df)
            df.to_csv(output_file)


if __name__ == '__main__':
    parser = ArgumentParser(description="Evaluates a 3D Unet model")
    parser.add_argument("-l", "--logdir", help="Directory where the Keras models are stored",
                        default="/home/snarduzz/Models")
    parser.add_argument('-i', '--input_channels', nargs='+', action="append", help='<Required> Set flag', default=None)
    parser.add_argument('-o', '--output_channels', nargs='+', action="append", help='<Required> Set flag', default=None)
    parser.add_argument('-r', '--root_data_folder', help="Root folder to replace", default="/home/snarduzz/Data")
    parser.add_argument('-b', '--backup_data_folder', help="Backup folder that replace root folder", default="/home/snarduzz/Data")

    args = parser.parse_args()
    channels_input = args.input_channels
    channels_output = args.output_channels
    logdir = os.path.expanduser(args.logdir)
    to_replace_dict = {args.root_data_folder: args.backup_data_folder}

    if channels_input is None:
        channels_input = ["TRACE", "T2"]
    else:
        channels_input = [x[0] for x in channels_input]

    if channels_output is None:
        channels_output = ["LESION"]
    else:
        channels_output = [x[0] for x in channels_output]

    print("INPUTS : {}".format(channels_input))
    print("OUTPUTS : {}".format(channels_output))

    df = None
    columns = ["model_name", "date", "iteration", "loss_function",
               "auc", "auc_mean", "auc_std",
               "dice", "dice_mean", "dice_std", "dice_thresh",
               "tversky", "tversky_mean", "tversky_std", "tversky_thresh",
               "weighted-dice",
               "ap", "f1-score", "f1_mean", "f1_std", "jaccard", "accuracy", "precision",
               "sensitivity", "specificity", "val_loss",
               "parameters_file"]

    if "checkpoints" not in os.listdir(logdir):
        print("No checkpoints found. Assuming this is a root folder of all models.")

        for x in os.listdir(logdir):
            model_dir = os.path.join(logdir, x)
            print("Evaluating {}...".format(x))
            evaluate_dir(model_dir, channels_input, channels_output, to_replace=to_replace_dict)
    else:
        evaluate_dir(logdir, channels_input, channels_output, to_replace=to_replace_dict)

