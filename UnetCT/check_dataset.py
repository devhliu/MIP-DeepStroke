import numpy as np
import os
from tqdm import tqdm
import nibabel as nb

from argparse import ArgumentParser
import json

if __name__ == '__main__':

    parser = ArgumentParser(description="Train a 3D Model unet")
    parser.add_argument("-p", "--path", help="Directory where to log the data for tensorboard",
                        default="/home/snarduzz/Data")

    args = parser.parse_args()
    directory = args.path

    folders = os.listdir(directory)
    folders = [x for x in folders if x in ["train","test","validation"]]

    dict_log = dict()

    if len(folders)==0:
        raise Exception("The root directory should contain train/test/validation sets")

    for f in folders:
        subdirectory = os.path.join(directory,f)
        files = os.listdir(subdirectory)
        files = [x for x in files if x.endswith(".nii")]
        print("Found {} .nii files. Checking....".format(len(files)))

        for file in tqdm(files):
            img = nb.load(file).get_data()
            max_img = np.max(img)
            min_img = np.min(img)
            if max_img>1 or min_img<-1:
                dict_log[file] = (min_img,max_img)

    # Save report
    json_file = os.path.join(directory,"report.json")
    with open(json_file, 'w') as fp:
        json.dump(dict_log, fp)

    print("Found {} incorrect files. Report is available here : {}".format(len(dict_log), json_file))
