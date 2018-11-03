from argparse import ArgumentParser
from utils import create_if_not_exists, splits_sets, split_train_test_val, normalize_numpy
from image_processing import load_data_atlas_for_patient, create_patches_from_images, create_extra_patches_from_list, preprocess_image
from image_processing import to_patches_3d, to_patches_3d_augmented_with_1
import numpy as np
import nibabel as nb
import os
from tqdm import tqdm
import pickle
import datetime

def _save_patches(patch_list, save_path, subject, type, extra=False):
    for i in range(len(patch_list)):
        patch = nb.Nifti1Image(patch_list[i], np.eye(4))
        saving_path = create_if_not_exists(save_path)
        patch_save_path = create_if_not_exists(os.path.join(saving_path, type))
        s = "{}_{}-{}.nii"
        if extra:
            s = "{}_{}-{}-extra.nii"
        nb.save(patch, os.path.join(patch_save_path, s.format(type, subject, i)))

def load_data_for_patient(patient_path):
    p = os.path.basename(patient_path)
    lesion = nb.load(os.path.join(patient_path, "Neuro_Cerebrale_64Ch", "wcoreg_VOI_lesion_{}.nii".format(p))).get_data()
    MTT = nb.load(os.path.join(patient_path, "Ct2_Cerebrale", "wcoreg_RAPID_MTT_{}.nii".format(p))).get_data()
    Tmax = nb.load(os.path.join(patient_path, "Ct2_Cerebrale", "wcoreg_RAPID_Tmax_{}.nii".format(p))).get_data()
    CBF = nb.load(os.path.join(patient_path, "Ct2_Cerebrale", "wcoreg_RAPID_rCBF_{}.nii".format(p))).get_data()
    CBV = nb.load(os.path.join(patient_path, "Ct2_Cerebrale", "wcoreg_RAPID_rCBV_{}.nii".format(p))).get_data()
    T2 = nb.load(os.path.join(patient_path, "Neuro_Cerebrale_64Ch", "wcoreg_t2_tse_tra_{}.nii".format(p))).get_data()

    return MTT, CBF, CBV, Tmax, T2, lesion


def _create_data_for_patients(dataset, save_path, dataset_type="train", ratio_extra=0.3):
    print("Creating dataset {} : ".format(dataset_type))
    # Create patches for train
    for patient_path in tqdm(dataset):
        subject = os.path.basename(patient_path)
        # load all data
        try:
            MTT, CBF, CBV, Tmax, T2, lesion = load_data_for_patient(patient_path)
        except Exception as e:
            print("Error while reading patient {}".format(patient_path))
            print(str(e))
            continue
        
       # preprocess data (normalize data)
        MTT = preprocess_image(MTT)
        CBF = preprocess_image(CBF)
        CBV = preprocess_image(CBV)
        Tmax = preprocess_image(Tmax)
        lesion = preprocess_image(lesion)
        T2 = preprocess_image(T2)

        # create patches, by doing overlap in case of training set
        is_train = (dataset_type == 'train')
        MTT_patches = create_patches_from_images(MTT, patch_size, augment=is_train)
        CBF_patches = create_patches_from_images(CBF, patch_size, augment=is_train)
        CBV_patches = create_patches_from_images(CBV, patch_size, augment=is_train)
        Tmax_patches = create_patches_from_images(Tmax, patch_size, augment=is_train)
        lesion_patches = create_patches_from_images(lesion, patch_size, augment=is_train)
        T2_patches = create_patches_from_images(T2, patch_size, augment=is_train)

        _save_patches(MTT_patches, save_path, subject=subject, type="MTT")
        _save_patches(CBF_patches, save_path, subject=subject, type="CBF")
        _save_patches(CBV_patches, save_path, subject=subject, type="CBV")
        _save_patches(Tmax_patches, save_path, subject=subject, type="Tmax")
        _save_patches(lesion_patches, save_path, subject=subject, type="lesion")
        _save_patches(T2_patches, save_path, subject=subject, type="T2")

        # create extra patches
        if is_train:
            number_extra = int(ratio_extra*len(MTT_patches))
            MTT_extra, CBF_extra, CBV_extra, Tmax_extra, lesion_extra, T2_extra = create_extra_patches_from_list(
                                                                            [MTT,
                                                                            CBF,
                                                                            CBV,
                                                                            Tmax,
                                                                            T2],
                                                                            lesion,
                                                                            patch_size,
                                                                            limit=number_extra)

            _save_patches(MTT_extra, save_path, subject=subject, type="MTT", extra=True)
            _save_patches(CBF_extra, save_path, subject=subject, type="CBF", extra=True)
            _save_patches(CBV_extra, save_path, subject=subject, type="CBV", extra=True)
            _save_patches(Tmax_extra, save_path, subject=subject, type="Tmax", extra=True)
            _save_patches(lesion_extra, save_path, subject=subject, type="lesion", extra=True)
            _save_patches(T2_extra, save_path, subject=subject, type="T2", extra=True)


if __name__ == '__main__':
    parser = ArgumentParser(description="Create a dataset from folder")

    parser.add_argument("-d", "--data_path", help="Path to data folder",
                        default="/home/snarduzz/Data/preprocessed_original_masked")
    parser.add_argument("-s", "--save_path", help="Path where to save patches", default="/home/snarduzz/Data")
    parser.add_argument("-p", "--patch_size", help="Patch size", type=int, default=32)
    parser.add_argument("-f", "--setfile", help="File where the distribution of patient is stored", default=None)

    args = parser.parse_args()

    date = datetime.datetime.now().strftime("%d%m%y-%H%M")

    patch_size = [args.patch_size, args.patch_size, args.patch_size]
    string_patches = "x".join([str(x) for x in patch_size])

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
        train, test, val, _, _, _ = split_train_test_val(patients_paths, ["" for x in range(len(patients_paths))], ratios=ratios)

        dict_sets = {"train":train,
                     "test":test,
                     "val":val}
        filename = os.path.join(dataset_data_path, "sets_{}".format(date))

        with open('{}.pickle'.format(filename), 'wb') as handle:
            pickle.dump(dict_sets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(args.setfile, 'rb') as handle:
            dict_sets = pickle.load(handle)
        #ratios
        total = len(dict_sets["train"])+len(dict_sets["test"])+len(dict_sets["val"])
        train = dict_sets["train"]
        test = dict_sets["test"]
        val = dict_sets["val"]
        ratios = [len(train)/total, len(test)/total, len(val)/total]

    print("------ Total :", len(patients_paths), "patients ------")
    print(len(train), "patients will be used for train ({}%)".format(ratios[0]*100))
    print(len(test), "patients will be used for test ({}%)".format(ratios[1]*100))
    print(len(val), "patients will be used for validation ({}%)".format(ratios[2]*100))

    # Create folders to save the data
    train_path = create_if_not_exists(os.path.join(save_path, "train"))
    test_path = create_if_not_exists(os.path.join(save_path, "test"))
    validation_path = create_if_not_exists(os.path.join(save_path, "validation"))

    _create_data_for_patients(train, train_path, dataset_type="train")
    _create_data_for_patients(test, test_path, dataset_type="test")
    _create_data_for_patients(val, validation_path, dataset_type="validation")

