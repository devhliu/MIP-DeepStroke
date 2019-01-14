import os
import nibabel as nb
from argparse import ArgumentParser
import numpy as np
import random

def clean_lesions(lesion_path, pos=1.0, neg=0.0):
    files = os.listdir(lesion_path)
    for f in files:
        filepath = os.path.join(lesion_path, f)
        img = nb.load(filepath).get_data()
        if np.sum(img)>0:
            r = np.random.uniform()
            if r>pos:
                os.remove(filepath)
        else:
            r = np.random.uniform()
            if r > neg:
                os.remove(filepath)

def clean_folders(path_lesion, modalities=["MTT","Tmax","CVF","CBF","T2","background"]):
    kept_lesion_files = os.listdir(path_lesion)

    for m in modalities:
        folder_path = path_lesion.replace("LESION",m)
        if not os.path.exists(folder_path):
            continue
        files = os.listdir(folder_path)
        for f in files:
            filepath = os.path.join(folder_path, f)
            corresponding_lesion_path = os.path.basename(filepath.replace(m, "LESION"))
            if corresponding_lesion_path not in kept_lesion_files:
                os.remove(filepath)


if __name__ == '__main__':
    parser = ArgumentParser(description="Create a dataset from folder")

    parser.add_argument("-d", "--data_path", help="Path to data folder",
                        default="/home/snarduzz/Data/")
    parser.add_argument("-pos", "--pos", help="Positive ratio to keep", type=float,
                        default=1.0)
    parser.add_argument("-neg", "--neg", help="Negative ratio to keep", type=float,
                        default=0.0)

    args = parser.parse_args()
    path = args.data_path
    pos = args.pos
    neg = args.neg

    print("Keeping {}% of positive samples".format(pos*100.0))
    print("Keeping {}% of negative samples".format(neg*100.0))

    folders = os.listdir(path)
    for f in folders:
        if f.lower() not in [x.lower() for x in ["TRACE","MTT","Tmax","CBF","CBV","LESION","background","T2"]]:
            raise Exception("Folder not allowed")

    clean_lesions(os.path.join(path, "LESION"))
    clean_folders(os.path.join(path, "LESION"), modalities=["TRACE","MTT","Tmax","CVF","CBF","T2","background"])