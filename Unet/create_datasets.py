from argparse import ArgumentParser
from utils import create_if_not_exists, splits_sets, split_train_test_val
from image_processing import load_data_atlas_for_patient, create_patches_from_images, create_extra_patches
import numpy as np
import nibabel as nb
import os
from tqdm import tqdm


def _save_patches(patch_list, save_path, subject, type, extra=False):
    for i in range(len(patch_list)):
        patch = nb.Nifti1Image(patch_list[i], np.eye(4))
        saving_path = create_if_not_exists(save_path)
        patch_save_path = create_if_not_exists(os.path.join(saving_path, type))
        s = "{}_{}-{}.nii"
        if extra:
            s = "{}_{}-{}-extra.nii"
        nb.save(patch, os.path.join(patch_save_path, s.format(type, subject, i)))


def _create_data_for_patients(dataset, save_path, dataset_type="train"):
    print("Creating dataset {} : ".format(dataset_type))
    # Create patches for train
    for patient_path in tqdm(dataset):
        subject = os.path.basename(patient_path)
        # load all data
        brain, lesion = load_data_atlas_for_patient(patient_path)

        # create patches
        input_patches = create_patches_from_images(brain, patch_size)
        label_patches = create_patches_from_images(lesion, patch_size)
        _save_patches(input_patches, save_path, subject=subject, type="input")
        _save_patches(label_patches, save_path, subject=subject, type="mask")

        # create extra patches
        if dataset_type=="train":
            ratio_extra = 0.3
            number_extra = int(ratio_extra*len(input_patches))
            input_patches_extra, lesion_patches_extra = create_extra_patches(brain, lesion, patch_size, limit=number_extra)
            _save_patches(input_patches_extra, save_path, subject=subject, type="input", extra=True)
            _save_patches(lesion_patches_extra, save_path, subject=subject, type="mask", extra=True)


if __name__ == '__main__':
    parser = ArgumentParser(description="Train a 3D Model unet")
    parser.add_argument("-l", "--logdir", help="Directory where to log the data for tensorboard",
                        default="/home/snarduzz/")

    parser.add_argument("-d", "--data_path", help="Path to data folder (ATLAS)",
                        default="/home/simon/Datasets/Stroke_DeepLearning_ATLASdataset")
    parser.add_argument("-s", "--save_path", help="Path where to save patches", default="/home/simon/Datasets")
    parser.add_argument("-p", "--patch_size", help="Patch size", type=int, default=32)
    args = parser.parse_args()

    patch_size = [args.patch_size, args.patch_size, args.patch_size]
    string_patches = "x".join([str(x) for x in patch_size])

    dataset_path = create_if_not_exists(args.save_path)
    dataset_data_path = create_if_not_exists(os.path.join(dataset_path, "Data"))
    save_path = create_if_not_exists(os.path.join(dataset_data_path, string_patches))

    # Load patients paths
    sites = [os.path.join(args.data_path, x) for x in os.listdir(args.data_path)]
    patients_paths = []
    for site in sites:
        patients = os.listdir(site)
        patients_paths = patients_paths+[os.path.join(site, x) for x in patients]

    patients_paths = patients_paths[:10]

    # Split set of patients into train, test and val sets
    ratios = [0.7, 0.2, 0.1]
    train, test, val, _, _, _ = split_train_test_val(patients_paths, ["" for x in range(len(patients_paths))], ratios=ratios)
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

