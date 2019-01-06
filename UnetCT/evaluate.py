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


def predict(test_folder, model, channels_input=["T2"], channels_output=["lesion"]):
    input_option = channels_input[0]
    input_files = [os.path.join(test_folder, input_option, x) for x in os.listdir(os.path.join(test_folder, input_option))]

    s = nb.load(input_files[0]).get_data().size

    patient_list = list(set([re.search('%s(.*)%s' % ("{}_".format(input_option), "-"), x).group(1) for x in input_files]))

    files_per_patients = len([x for x in input_files if patient_list[0] in x])

    list_y = np.empty(len(patient_list)*files_per_patients*s)
    list_y_pred = np.empty(len(patient_list)*files_per_patients*s)

    aucs = []
    dices = []
    precisions = []
    recalls = []

    aucs_thresh = []
    dices_thresh = []
    precisions_thresh = []
    recalls_thresh = []

    for i in tqdm(range(len(patient_list))):
        p = patient_list[i]
        y_true, y_pred = predict_patient(p, input_files, model, channels_input, channels_output)

        y_true = y_true.flatten()
        y_pred = y_pred.flatten()


        # without threshold
        auc = metrics.roc_auc_score(y_true, y_pred)
        dice = dice_score(y_true=y_true, y_pred=y_pred)
        precision = metrics.precision_score(y_true=y_true, y_pred=y_pred)
        recall = metrics.recall_score(y_true=y_true, y_pred=y_pred)

        aucs.append(auc)
        dices.append(dice)
        precisions.append(precision)
        recalls.append(recall)

        # with threshold
        # compute threshold
        y_pred_thresh = y_pred.copy()
        y_pred_thresh[y_pred_thresh < 0.5] = 0
        y_pred_thresh[y_pred_thresh >= 0.5] = 1

        auc_t = metrics.roc_auc_score(y_true, y_pred_thresh)
        dice_t = dice_score(y_true=y_true, y_pred=y_pred_thresh)
        precision_t = metrics.precision_score(y_true=y_true, y_pred=y_pred_thresh)
        recall_t = metrics.recall_score(y_true=y_true, y_pred=y_pred_thresh)

        aucs_thresh.append(auc_t)
        dices_thresh.append(dice_t)
        precisions_thresh.append(precision_t)
        recalls_thresh.append(recall_t)


    dict_scores = {}

    # without threshold
    dict_scores["AUC"] = np.mean(aucs)
    dict_scores["AUC_std"] = np.std(aucs)
    dict_scores["DSC"] = np.mean(dices)
    dict_scores["DSC_std"] = np.std(dices)
    dict_scores["Recall"] = np.mean(recalls)
    dict_scores["Recall_std"] = np.std(recalls)
    dict_scores["Precision"] = np.mean(precisions)
    dict_scores["Precision_std"] = np.std(precisions)
    # with threshold
    dict_scores["thresh_AUC"] = np.mean(aucs_thresh)
    dict_scores["thresh_AUC_std"] = np.std(aucs_thresh)
    dict_scores["thresh_DSC"] = np.mean(dices_thresh)
    dict_scores["thresh_DSC_std"] = np.std(dices_thresh)
    dict_scores["thresh_Recall"] = np.mean(recalls_thresh)
    dict_scores["thresh_Recall_std"] = np.std(recalls_thresh)
    dict_scores["thresh_Precision"] = np.mean(precisions_thresh)
    dict_scores["thresh_Precision_std"] = np.std(precisions_thresh)


    return dict_scores


def evaluate_dir(logdir, to_replace={"/home/snarduzz/Data":"/home/snarduzz/Data"}):
    checkpoints_folder = os.path.join(logdir, "checkpoints")
    parameters_file = os.path.join(logdir, "parameters.json")
    output_file = os.path.join(logdir, "evaluation.csv")
    date = os.path.basename(logdir)
    # load parameters
    print("Loading parameters from : " + parameters_file)
    with open(parameters_file, 'r') as fp:
        parameters = json.load(fp)

    path_replaced = parameters["data_path"]
    channels_input = parameters["inputs"]
    channels_output = parameters["targets"]

    print("INPUTS : {}".format(channels_input))
    print("OUTPUTS : {}".format(channels_output))

    if not os.path.exists(path_replaced):
        # try by replacing the value
        for k in to_replace.keys():
            path_replaced = path_replaced.replace(k, to_replace[k])

    # if still not valid
    if not os.path.exists(path_replaced):
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

            columns_meta = ["date", "model_name", "iteration", "val_loss", "loss_function"]
            columns_metrics = ["AUC", "AUC_std", "DSC", "DSC_std", "Recall", "Recall_std", "Precision", "Precision_std"]
            columns_metrics_tresh = ["tresh_"+x for x in columns_metrics]

            columns = columns_meta + columns_metrics + columns_metrics_tresh

            # Append line to CSV by keeping only relevant columns
            df = df[columns]
            new_df = pd.DataFrame(dict_scores, index=[0])
            df = df.append(new_df)
            df.to_csv(output_file)


if __name__ == '__main__':
    parser = ArgumentParser(description="Evaluates a 3D Unet model")
    parser.add_argument("-l", "--logdir", help="Directory where the Keras models are stored",
                        default="/home/snarduzz/Models")
    parser.add_argument('-r', '--root_data_folder', help="Root folder to replace", default="/home/snarduzz/Data")
    parser.add_argument('-b', '--backup_data_folder', help="Backup folder that replace root folder", default="/home/snarduzz/Data")

    args = parser.parse_args()
    logdir = os.path.expanduser(args.logdir)
    to_replace_dict = {args.root_data_folder: args.backup_data_folder}

    df = None

    if "checkpoints" not in os.listdir(logdir):
        print("No checkpoints found. Assuming this is a root folder of all models.")

        output_file = os.path.join(logdir, "total_evaluation.csv")
        list_frames = []

        for x in sorted(os.listdir(logdir)):
            model_dir = os.path.join(logdir, x)
            print("Evaluating {}...".format(x))
            evaluate_dir(model_dir, to_replace=to_replace_dict)

            #load recently created evaluation file
            evaluation_file = os.path.join(model_dir, "evaluation.csv")
            df = pd.read_csv(evaluation_file, index_col=None, header=0)
            list_frames.append(df)

        frame = pd.concat(list_frames, axis=0, ignore_index=True)

    else:
        evaluate_dir(logdir, to_replace=to_replace_dict)

