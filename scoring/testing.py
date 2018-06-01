import numpy as np
import nibabel as nb
from Unet.predict import predict_patch
import os
from argparse import ArgumentParser
from tqdm import tqdm
from keras.models import load_model
from Unet.metrics import *
import sklearn.metrics as metrics


def predict(test_folder, model, file):
    input_files = [os.path.join(test_folder,"input",x) for x in os.listdir(os.path.join(test_folder,"input"))]

    s = nb.load(input_files[0]).get_data().size
    list_y = np.empty(len(input_files)*s)
    list_y_pred = np.empty(len(input_files)*s)

    for i in tqdm(range(len(input_files))):
        if(i>8000):
            continue
        x_file = input_files[i]
        y_file = input_files[i].replace("input","mask")

        x = nb.load(x_file).get_data()
        y = nb.load(y_file).get_data().astype(np.int8)

        # Predict one patch
        #y_pred = predict_patch(x, model=model)
        y_pred = np.zeros(x.shape)

        list_y[i:i+s] = y.flatten()
        list_y_pred[i:i+s]=y_pred.flatten()

    fpr, tpr, thresholds = metrics.roc_curve(list_y, list_y_pred, pos_label=1)
    AUC = metrics.auc(fpr, tpr)
    print("AUC:{}".format(AUC))


if __name__ == '__main__':
    parser = ArgumentParser(description="Train a 3D Model unet")
    parser.add_argument("-l", "--logdir", help="Directory where the Keras models are stored",
                        default="/home/simon/models")

    parser.add_argument("-d", "--data_path", help="Path to test data folder",
                        default="/home/simon/Datasets/Data/32x32x32/test/")
    parser.add_argument("-o", "--output_file", help="Name of the CSV where the data will be stored",
                        default="/home/simon/models/results.csv")

    args = parser.parse_args()

    output_file = args.output_file
    data_path = args.data_path
    logdir = args.logdir

    runs = [os.path.join(logdir, x) for x in os.listdir(logdir)]
    for run in runs:
        checkpoints_folder = os.path.join(run, "checkpoints")
        if os.path.exists(checkpoints_folder):

            checkpoints = [os.path.join(checkpoints_folder, x) for x in os.listdir(checkpoints_folder)]

            # For all checkpoints, evaluate.
            for ckpt in tqdm(checkpoints,desc=run):
                model = load_model(ckpt,custom_objects={"dice_coefficient":dice_coefficient,
                                                        "dice_coefficient_loss": dice_coefficient_loss,
                                                        "weighted_dice_coefficient_loss":weighted_dice_coefficient_loss,
                                                        "weighted_dice_coefficient":weighted_dice_coefficient,
                                                        "tversky_loss": tversky_loss,
                                                        })

                predict(data_path, model, output_file)
        else:
            if("logs" in os.listdir(run)):
                os.removedirs(os.path.join(run, "logs"))
            os.removedirs(run)


