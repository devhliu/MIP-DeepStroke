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
import re

def specificity(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity


def predict_patient(patient_id, list_files, model, channels_input=["MTT", "Tmax", "CBV", "CBF"], channels_output=["lesion"]):
    patient_patches_names = [x for x in list_files if patient_id in x]

    ys_true = []
    ys_pred = []

    input_option = channels_input[0]
    for i in tqdm(range(len(patient_patches_names))):
        inputs_patches = []
        for c in channels_input:
            x_file = patient_patches_names[i].replace(input_option,c)
            x = nb.load(x_file).get_data()
            inputs_patches.append(x)
        output_patches = []
        for c in channels_output:
            y_file = patient_patches_names[i].replace(input_option, c)
            y = nb.load(y_file).get_data().astype(np.int8)
            output_patches.append(y)

        inputs_patches = np.array(inputs_patches)
        output_patches = np.array(output_patches)

        # Predict one patch
        y_pred = predict_patch(inputs_patches, model=model)

        ys_true += list(output_patches.flatten())
        ys_pred += list(y_pred.flatten())

    return np.array(ys_true), np.array(ys_pred)


def predict(test_folder, model, maxsize=None, channels_input=["MTT", "Tmax", "CBV", "CBF"], channels_output=["lesion"]):
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

    for i in tqdm(range(len(patient_list))):
        p = patient_list[i]
        y_true, y_pred = predict_patient(p, input_files, model, channels_input, channels_output)

        auc = metrics.roc_auc_score(y_true, y_pred)

        # Treshold
        y_pred_t = y_pred.copy()
        y_pred_t[y_pred_t < 0.5] = 0
        y_pred_t[y_pred_t >= 0.5] = 1
        f1 = metrics.f1_score(y_true, y_pred_t)

        aucs.append(auc)
        f1_scores.append(f1)

        list_y[i*len(y_true):(i+1)*len(y_true)] = y_true
        list_y_pred[i * len(y_pred):(i + 1) * len(y_pred)] = y_pred

    y_true = list_y
    y_pred = list_y_pred

    # AUC without threshold
    auc = metrics.roc_auc_score(y_true, y_pred)
    dict_scores = {"auc": auc}

    # AUC per patient with std
    dict_scores["auc_mean"] = np.mean(np.array(aucs))
    dict_scores["auc_std"] = np.std(np.array(aucs))
    dict_scores["f1_mean"] = np.mean(np.array(f1_scores))
    dict_scores["f1_std"] = np.std(np.array(f1_scores))

    functions = {"ap":metrics.average_precision_score,
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
        try:
            score = functions[k](y_true, y_pred)
        except Exception as e:
            print("error computing k")
            print(e)
            score = 0
        dict_scores[k] = score

    return dict_scores


if __name__ == '__main__':
    parser = ArgumentParser(description="Train a 3D Model unet")
    parser.add_argument("-l", "--logdir", help="Directory where the Keras models are stored",
                        default="/home/simon/models")

    parser.add_argument("-d", "--data_path", help="Path to test data folder",
                        default="/home/simon/Datasets/Data/32x32x32/test/")
    parser.add_argument("-o", "--output_file", help="Name of the CSV where the data will be stored",
                        type=str, default="/home/simon/models/results-32.csv")
    parser.add_argument('-ic', '--input_channels', nargs='+', help='<Required> Set flag', required=True, default=None)
    parser.add_argument('-oc', '--output_channels', nargs='+', help='<Required> Set flag', required=True, default=None)

    args = parser.parse_args()
    channels_input = args.input_channels
    channels_output = args.output_channels

    if channels_input is None:
        channels_input = ["CBV", "CBF", "MTT", "Tmax"]
    if channels_output is None:
        channels_output = ["lesion"]

    output_file = args.output_file
    data_path = args.data_path
    logdir = args.logdir

    df = None
    columns = ["auc", "auc_mean", "auc_std", "ap", "f1-score", "f1_mean", "f1_std", "jaccard", "accuracy", "precision",
               "sensitivity", "specificity", "model_name", "run", "date", "iteration", "val_acc"]

    # Create DF if not exists
    if not os.path.exists(output_file):
        df = pd.DataFrame(columns=columns).reset_index()
        df.to_csv(output_file)


    runs = [os.path.join(logdir, x) for x in os.listdir(logdir) if os.path.isdir(os.path.join(logdir,x))]
    for run in runs:
        checkpoints_folder = os.path.join(run, "checkpoints")
        if os.path.exists(checkpoints_folder):

            checkpoints = [os.path.join(checkpoints_folder, x) for x in os.listdir(checkpoints_folder)]
            if(len(checkpoints)<1):
                continue
            # For all checkpoints, evaluate.
            for ckpt in tqdm(checkpoints, desc=run):

                # put extra information
                model_name = os.path.basename(ckpt)
                str_info = model_name.replace("model.", "").replace("_", "").replace(".hdf5", "").split("-")
                str_info = [x for x in str_info if len(x) > 0]
                it, val_acc = float(str_info[0]), float(str_info[1])

                # Load CSV and append line
                df = pd.read_csv(output_file, header=0)

                # Check that the model was not already tested
                if run in df["run"].values and model_name in df["model_name"].values:
                    print(run, model_name, "already tested.")
                    continue

                model = load_model(ckpt, custom_objects={"dice_coefficient" : dice_coefficient,
                                                        "dice_coefficient_loss" : dice_coefficient_loss,
                                                        "weighted_dice_coefficient_loss" : weighted_dice_coefficient_loss,
                                                        "weighted_dice_coefficient" : weighted_dice_coefficient,
                                                        "tversky_loss" : tversky_loss,
                                                        "tversky_coeff": tversky_coeff
                                                        })
                dict_scores = {}
                try:
                    dict_scores = predict(data_path, model, channels_input, channels_output)
                except Exception as e:
                    print("Error while predicting model {}. Try with another image patch size.".format(ckpt))
                    print(e)
                    continue

                dict_scores["model_name"] = model_name
                dict_scores["run"] = run
                run_date = os.path.basename(run)
                format = "%Y%m%d_%H-%M-%S"
                dict_scores["date"] = datetime.strptime(run_date, format)
                dict_scores["iteration"] = it
                dict_scores["val_acc"] = val_acc

                # Append line to CSV by keeping only relevant columns
                df = df[list(dict_scores.keys())]
                new_df = pd.DataFrame(dict_scores, index=[0])
                df = df.append(new_df)
                df.to_csv(output_file)


        else:
            if("logs" in os.listdir(run)):
                os.removedirs(os.path.join(run, "logs"))
            os.removedirs(run)


