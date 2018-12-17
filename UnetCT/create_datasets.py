from argparse import ArgumentParser
from utils import create_if_not_exists, splits_sets, split_train_test_val, normalize_numpy
from image_processing import load_data_atlas_for_patient, create_patches_from_images, create_extra_patches_from_list, preprocess_image
from image_processing import to_patches_3d, to_patches_3d_augmented_with_1
from multiprocessing import Process
import numpy as np
import nibabel as nb
import os
from tqdm import tqdm
import pickle
import datetime
import json

def _save_patches(patch_list, save_path, subject, type, extra=False):
    for i in range(len(patch_list)):
        patch = nb.Nifti1Image(patch_list[i], np.eye(4))
        saving_path = create_if_not_exists(save_path)
        patch_save_path = create_if_not_exists(os.path.join(saving_path, type))
        s = "{}_{}-{}.nii"
        if extra:
            s = "{}_{}-{}-extra.nii"
        nb.save(patch, os.path.join(patch_save_path, s.format(type, subject, i)))

def load_data_for_patient(patient_path, stage="wcoreg_", modalities=["TRACE","T2","LESION"]):
    p = os.path.basename(patient_path)
    dict_modalities_path = {
        "MTT" : os.path.join(patient_path, "Ct2_Cerebrale", "{}RAPID_MTT_{}.nii".format(stage, p)),
        "TMAX": os.path.join(patient_path, "Ct2_Cerebrale", "{}RAPID_Tmax_{}.nii".format(stage, p)),
        "CBF": os.path.join(patient_path, "Ct2_Cerebrale", "{}RAPID_rCBF_{}.nii".format(stage, p)),
        "CBV": os.path.join(patient_path, "Ct2_Cerebrale", "{}RAPID_rCBV_{}.nii".format(stage, p)),
        "T2": os.path.join(patient_path, "Neuro_Cerebrale_64Ch", "{}t2_tse_tra_{}.nii".format(stage, p)),
        "TRACE": os.path.join(patient_path, "Neuro_Cerebrale_64Ch", "{}diff_trace_tra_TRACEW_{}.nii".format(stage, p)),
        "LESION": os.path.join(patient_path, "Neuro_Cerebrale_64Ch", "{}VOI_lesion_{}.nii".format(stage, p))
    }

    returned_dict = dict()
    for modality in modalities:
        returned_dict[modality] = nb.load(dict_modalities_path[modality]).get_data()

    return returned_dict


def _create_data_for_patients(dataset, save_path, dataset_type="train", ratio_extra=0.3, preprocessing="standardize",
                              stage="wcoreg_", augment=False , mode="extend", modalities=["TRACE","T2","LESION"]):
    print("Creating dataset {} : ".format(dataset_type))
    # Create patches for train
    for patient_path in tqdm(dataset):
        subject = os.path.basename(patient_path)
        # load all data
        try:
            returned_dict = load_data_for_patient(patient_path, stage=stage)
        except Exception as e:
            print("Error while reading patient {}".format(patient_path))
            print(str(e))
            continue
        
       # preprocess data (normalize data)
        dict_preprocess = dict()
        for modality in modalities:
            if modality!="LESION" and modality!="BACKGROUND":
                image = preprocess_image(returned_dict[modality], preprocessing=preprocessing)
            else:
                image = preprocess_image(returned_dict[modality], preprocessing="normalize")
            dict_preprocess[modality] = image

        # create patches, by doing overlap in case of training set
        is_train = (dataset_type == 'train')
        image_patch = None
        for modality in modalities:
            image_patch = create_patches_from_images(dict_preprocess[modality], patch_size, augment=augment, mode=mode)
            _save_patches(image_patch, save_path, subject=subject, type=modality)

        # create extra patches
        if is_train and augment is True:
            number_extra = int(ratio_extra*len(image_patch)) #take last size of image_patches

            #print("Creating {}*{} = {} extra patches.".format(ratio_extra, len(MTT_patches), number_extra))
            sorted_keys = sorted([k for k,v in dict_preprocess.items() if k!="LESION"])
            list_images = [dict_preprocess[k] for k in sorted_keys]
            lesion = dict_preprocess["LESION"]

            #Get extra images for modalities-1 + lesion
            extra_images = create_extra_patches_from_list(list_images, lesion, patch_size, limit=number_extra)
            extra_patches = dict()
            for i in range(0,len(sorted_keys)):
                extra_patches[sorted_keys[i]] = extra_images[i]

            #Last element of the list contains the lesion
            extra_patches["LESION"] = extra_images[-1]

            for k, v in extra_patches.items():
                _save_patches(v, save_path, subject=subject, type=k, extra=True)


if __name__ == '__main__':
    parser = ArgumentParser(description="Create a dataset from folder")

    parser.add_argument("-d", "--data_path", help="Path to data folder",
                        default="/home/snarduzz/Data/preprocessed_original_masked")
    parser.add_argument("-s", "--save_path", help="Path where to save patches", default="/home/snarduzz/Data")
    parser.add_argument("-p", "--patch_size", help="Patch size",  nargs="*", default=32, type=int)
    parser.add_argument("-f", "--setfile", help="File where the distribution of patient is stored", default=None)
    parser.add_argument("-pre", "--preprocessing", help="Preprocessing method", default="standardize")
    parser.add_argument("-stage", "--stage", help="Stage of regstration : coreg_ or wcoreg_ or \"\"", default="wcoreg_")
    parser.add_argument("-a", "--augment", type=bool, default=False, help="Augment the dataset")
    parser.add_argument("-m", "--mode", type=str, default="extend", help="What to do in case of mismatch of size : crop or extend?")
    parser.add_argument("-o", "--modalities", type=str, default=["T2", "TRACE", "LESION"],
                        help="What modalitites to extract")
    args = parser.parse_args()

    patch_size = [x for x in args.patch_size]
    if len(patch_size)==1:
        patch_size = [patch_size[0], patch_size[0], patch_size[0]]

    date = datetime.datetime.now().strftime("%d%m%y-%H%M")

    string_patches = "x".join([str(x) for x in patch_size])
    stage = args.stage
    mode = args.mode
    modalities = args.modalities
    dataset_path = create_if_not_exists(args.save_path)
    dataset_data_path = create_if_not_exists(os.path.join(dataset_path, date))
    save_path = create_if_not_exists(os.path.join(dataset_data_path, string_patches))

    if(args.setfile is None):
        # Load patients paths
        patients_paths = [os.path.join(args.data_path, x) for x in os.listdir(args.data_path) if x.isdigit()]

        print("The following patients will be processed:")
        print("")
        print([os.path.basename(x) for x in patients_paths])
        print("")

        # Split set of patients into train, test and val sets
        ratios = [0.7, 0.2, 0.1]
        if len(patients_paths)<4:
            train = patients_paths
            test = []
            val = []
        else:
            train, test, val, _, _, _ = split_train_test_val(patients_paths, ["" for x in range(len(patients_paths))], ratios=ratios)

        dict_sets = {"train":train,
                     "test":test,
                     "validation":val}

        filename = os.path.join(dataset_data_path, "sets_{}.json".format(date))

        with open(filename, 'w') as fp:
            json.dump(dict_sets, fp, indent=4)
    else:
        with open(args.parameters, 'r') as fp:
            dict_sets = json.load(fp)

    #ratios
    total = len(dict_sets["train"])+len(dict_sets["test"])+len(dict_sets["validation"])
    train = dict_sets["train"]
    test = dict_sets["test"]
    val = dict_sets["validation"]
    ratios = [len(train)/total, len(test)/total, len(val)/total]

    print("------ Total :", total, "patients ------")
    print(len(train), "patients will be used for train ({}%)".format(ratios[0]*100))
    print(len(test), "patients will be used for test ({}%)".format(ratios[1]*100))
    print(len(val), "patients will be used for validation ({}%)".format(ratios[2]*100))

    # Print dimensions
    print("------ Data dimensions -----------------")
    images_loaded = load_data_for_patient(train[0], stage=stage)
    shape = []
    for k,v in images_loaded.items():
        print("{} : {}".format(k, v.shape))

    print("") 

    # Create folders to save the data
    train_path = create_if_not_exists(os.path.join(save_path, "train"))
    test_path = create_if_not_exists(os.path.join(save_path, "test"))
    validation_path = create_if_not_exists(os.path.join(save_path, "validation"))

    _create_data_for_patients(train, train_path, dataset_type="train", preprocessing=args.preprocessing, stage=stage,
                              augment=args.augment, mode=mode, modalities=modalities)
    _create_data_for_patients(test, test_path, dataset_type="test", preprocessing=args.preprocessing, stage=stage,
                              augment=False, mode=mode, modalities=modalities)
    _create_data_for_patients(val, validation_path, dataset_type="validation", preprocessing=args.preprocessing,
                              stage=stage,augment=False, mode=mode, modalities=modalities)

