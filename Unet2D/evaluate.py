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
import losses
from create_datasets import load_data_for_patient
from predict import predict

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
    patient_patches_names = sorted([x for x in list_files if patient_id in x])
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


def predict_patient_from(path, model, patch_size=[512, 512], stage="rcoreg_", input_folders=["T2"], target_folders=["LESION"], return_original=False, batch_size=32):
    modalities = input_folders+target_folders
    dict_inputs = load_data_for_patient(path, stage=stage, modalities=modalities, preprocess=True)

    images_input = [dict_inputs[k] for k in input_folders]
    images_target = [dict_inputs[k] for k in target_folders]

    # Predict the output.
    predicted_image = predict(images_input, model, patch_size, batch_size=batch_size)

    if return_original:
        return images_target[0], predicted_image

    return predicted_image


def compute_scores(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    if len(y_true) != len(y_pred):
        raise Exception("Cannot evaluate scores because original and prediction images does not match in shapes.")

    auc = metrics.roc_auc_score(y_true, y_pred)
    dice = dice_score(y_true=y_true, y_pred=y_pred)

    # with threshold
    # compute threshold
    y_pred_thresh = y_pred.copy()
    y_pred_thresh[y_pred_thresh < 0.5] = 0
    y_pred_thresh[y_pred_thresh >= 0.5] = 1

    auc_t = metrics.roc_auc_score(y_true, y_pred_thresh)
    dice_t = dice_score(y_true=y_true, y_pred=y_pred_thresh)
    precision_t = metrics.precision_score(y_true=y_true, y_pred=y_pred_thresh)
    recall_t = metrics.recall_score(y_true=y_true, y_pred=y_pred_thresh)

    d = dict()

    d["auc"] = auc
    d["dice"] = dice
    # threshold
    d["auc_t"] = auc_t
    d["dice_t"] = dice_t
    d["precision_t"] = precision_t
    d["recall_t"] = recall_t

    return d


def evaluate_model(model, dataset_path, inputs, targets, stage, patch_size, decimals=4,
                   to_replace={"/home/snarduzz/Data":"/mnt/sda/Data"}, batch_size=32):

    dataset_path = substitue_path(dataset_path, to_replace=to_replace) # Make sure path exists
    set_file = os.path.join(dataset_path, "set_parameters.json")

    # Load json containing patients
    with open(set_file, 'r') as fp:
        set_distribution = json.load(fp)

    test_patients = set_distribution["test"]

    if(len(test_patients)<1):
        raise Exception("No tests patients found for {}".format(dataset_path))

    else:  # perform analysis

        aucs = []
        dices = []

        aucs_thresh = []
        dices_thresh = []
        precisions_thresh = []
        recalls_thresh = []

        for patient_path in tqdm(test_patients):
            patient_path = substitue_path(patient_path, to_replace=to_replace)
            if not os.path.exists(patient_path):
                raise Exception("Patient does not exists : {} \n Please check that the dataset is still available.".format(patient_path))
            y_true, y_pred = predict_patient_from(patient_path, model, patch_size=patch_size, stage=stage, input_folders=inputs,
                                     target_folders=targets, return_original=True, batch_size=batch_size)

            patient_scores = compute_scores(y_true,y_pred)

            aucs.append(patient_scores["auc"])
            dices.append(patient_scores["dice"])
            aucs_thresh.append(patient_scores["auc_t"])
            dices_thresh.append(patient_scores["dice_t"])
            precisions_thresh.append(patient_scores["precision_t"])
            recalls_thresh.append(patient_scores["recall_t"])

        dict_scores = {}

        # without threshold
        dict_scores["AUC"] = round(np.mean(aucs), decimals)
        dict_scores["AUC_std"] = round(np.std(aucs), decimals)
        dict_scores["DSC"] = round(np.mean(dices), decimals)
        dict_scores["DSC_std"] = round(np.std(dices), decimals)
        # with threshold
        dict_scores["thresh_AUC"] = round(np.mean(aucs_thresh), decimals)
        dict_scores["thresh_AUC_std"] = round(np.std(aucs_thresh), decimals)
        dict_scores["thresh_DSC"] = round(np.mean(dices_thresh), decimals)
        dict_scores["thresh_DSC_std"] = round(np.std(dices_thresh), decimals)
        dict_scores["thresh_Recall"] = round(np.mean(recalls_thresh), decimals)
        dict_scores["thresh_Recall_std"] = round(np.std(recalls_thresh), decimals)
        dict_scores["thresh_Precision"] = round(np.mean(precisions_thresh), decimals)
        dict_scores["thresh_Precision_std"] = round(np.std(precisions_thresh), decimals)

        return dict_scores


def substitue_path(path, to_replace={"/home/snarduzz/Data":"/mnt/sda/Data"}):

    if os.path.exists(path):
        if path.endswith("/"):
            path = path[:-1]
        return path

    if not os.path.exists(path):
        path_replaced = path
        # try by replacing the value
        for k in to_replace.keys():
            path_replaced = path_replaced.replace(k, to_replace[k])

        # if still not valid
        if not os.path.exists(path_replaced):
            raise Exception("Path to data {} not found.".format(path_replaced))

        if path_replaced.endswith("/"):
            path_replaced = path_replaced[:-1]

        return path_replaced


def evaluate_dir(logdir, to_replace={"/home/snarduzz/Data":"/home/snarduzz/Data"}, decimals = 4):
    checkpoints_folder = os.path.join(logdir, "checkpoints")
    parameters_file = os.path.join(logdir, "parameters.json")
    output_file = os.path.join(logdir, "evaluation.csv")
    date = os.path.basename(logdir)
    # load parameters
    print("Loading parameters from : " + parameters_file)
    with open(parameters_file, 'r') as fp:
        parameters = json.load(fp)

    # replace the paths
    path = parameters["data_path"]
    dataset_path = substitue_path(path, to_replace=to_replace)

    channels_input = parameters["inputs"]
    channels_output = parameters["targets"]
    print("INPUTS : {}".format(channels_input))
    print("OUTPUTS : {}".format(channels_output))

    # Create DF if not exists
    if not os.path.exists(output_file):

        columns_meta = ["date", "model_name", "iteration", "val_loss", "loss_function"]
        columns_metrics = ["AUC", "AUC_std", "DSC", "DSC_std", "Recall", "Recall_std", "Precision", "Precision_std"]
        columns_metrics_tresh = ["thresh_" + x for x in columns_metrics]

        columns = columns_meta + columns_metrics[:4] + columns_metrics_tresh # only take AUC and DICE w/o thresh

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

            alpha_value, beta_value = parameters["tversky_alpha-beta"]

            model = load_model(ckpt, custom_objects={"dice_coefficient": dice_coefficient,
                                                     "dice_coefficient_loss": dice_coefficient_loss,
                                                     "weighted_dice_coefficient_loss": weighted_dice_coefficient_loss,
                                                     "weighted_dice_coefficient": weighted_dice_coefficient,
                                                     "tversky_loss": losses.tversky_loss,
                                                     "tversky_coeff": losses.tversky_coeff,
                                                     "dice_loss":losses.dice_loss,
                                                     "dsc":losses.dsc,
                                                     "focal_tversky":losses.focal_tversky,
                                                     "jaccard_distance":losses.jaccard_distance,
                                                     "tp": losses.tp,
                                                     "tn":losses.tn,
                                                     "<lambda>": losses.get_tversky(alpha_value, beta_value),
                                                     "tversky": losses.get_tversky(alpha_value, beta_value)
                                                     })
            dict_scores = {}
            try:
                stage = parameters["stage"]
                patch_size = [int(x) for x in os.path.basename(parameters["data_path"]).split("x")]
                print("patch size : {}".format(patch_size))
                batch_size = parameters["batch_size"]
                dict_scores = evaluate_model(model, dataset_path, channels_input, channels_output, stage, patch_size, decimals=decimals,
                                             to_replace=to_replace, batch_size=batch_size)
            except Exception as e:
                print(e)
                print("Error while predicting model {}. Try with another image patch size.".format(ckpt))
                traceback.print_exc()
                continue

            dict_scores["model_name"] = model_name
            dict_scores["date"] = date
            dict_scores["iteration"] = iteration
            dict_scores["val_loss"] = loss_value
            dict_scores["loss_function"] = parameters["loss_function"]

            # Append line to CSV by keeping only relevant columns
            df = df[columns]
            new_df = pd.DataFrame(dict_scores, index=[0])
            df = df.append(new_df, sort=False)
            df.to_csv(output_file)


if __name__ == '__main__':
    parser = ArgumentParser(description="Evaluates a 3D Unet model")
    parser.add_argument("-l", "--logdir", help="Directory where the Keras models are stored",
                        default="/home/snarduzz/Models")
    parser.add_argument('-r', '--root_data_folder', help="Root folder to replace", default="/home/snarduzz/Data")
    parser.add_argument('-b', '--backup_data_folder', help="Backup folder that replace root folder", default="/home/snarduzz/Data")
    parser.add_argument('-d', '--decimals', help="Decimals to round up results",
                        default=4)

    args = parser.parse_args()
    decimals = args.decimals
    logdir = os.path.expanduser(args.logdir)
    to_replace_dict = {args.root_data_folder: args.backup_data_folder}

    df = None

    if "checkpoints" not in os.listdir(logdir):
        print("No checkpoints found. Assuming this is a root folder of all models.")

        output_file = os.path.join(logdir, "total_evaluation.csv")
        list_frames = []

        for x in sorted(os.listdir(logdir)):
            model_dir = os.path.join(logdir, x)
            if not os.path.isdir(model_dir):
               print("Not a folder : {}".format(model_dir))
               continue
            print("Evaluating {}...".format(x))
            evaluate_dir(model_dir, to_replace=to_replace_dict, decimals=decimals)

            #load recently created evaluation file
            evaluation_file = os.path.join(model_dir, "evaluation.csv")
            df = pd.read_csv(evaluation_file, index_col=None, header=0)
            list_frames.append(df)

        frame = pd.concat(list_frames, axis=0, ignore_index=True, sort=False)
        frame.to_csv(output_file)

    else:
        evaluate_dir(logdir, to_replace=to_replace_dict, decimals=decimals)

