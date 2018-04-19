import os
import nibabel as nb
import numpy as np
import progressbar
from utils import create_if_not_exists
from sklearn.model_selection import train_test_split
from shutil import copyfile

def load_data_atlas_from_site(site_path):
    list_x = []
    list_y = []
    patients = os.listdir(site_path)
    with progressbar.ProgressBar(max_value=len(patients)) as bar:
        i = 0
        for patient in patients:
            bar.update(i)
            for t in os.listdir(os.path.join(site_path, patient)):
                if not t == "t01" or patient == "031916":
                    # print("Found folder {} instead of T01 for patient {}".format(t, patient))
                    continue
                brain_structure_path = os.path.join(site_path, patient, t, "output.nii")
                lesion_path = os.path.join(site_path, patient, t, "{}_LesionSmooth_stx.nii".format(patient))
                try:
                    brain_structure = nb.load(brain_structure_path).get_data()
                    lesion = nb.load(lesion_path).get_data()
                except Exception as e:
                    print("Error reading patient {} ({})".format(patient, os.path.basename(site_path)))
                    print(e.with_traceback())
                    continue

                list_x.append(brain_structure)
                list_y.append(lesion)
            i = i + 1

    return list_x, list_y


def create_patches_from_images(numpy_image, patch_size, mode="extend"):
    shape = numpy_image.shape
    missing = np.array([patch_size[i]-(shape[i]%patch_size[i]) for i in range(len(patch_size))])
    numpy_image_padded = np.zeros(numpy_image.shape+missing)

    if mode is "extend":
            numpy_image_padded[:,:,:] = np.pad(numpy_image[:,:,:], [(0, missing[0]), (0, missing[1]), (0, missing[2])],
                                               mode="constant", constant_values=0)

    shape = numpy_image_padded.shape
    dimension_size = np.array(np.ceil(np.array(numpy_image.shape[:3]) / patch_size), dtype=np.int64)
    xMax = dimension_size[0]
    yMax = dimension_size[1]
    zMax = dimension_size[2]
    patches = [0 for x in range(xMax*yMax*zMax)]

    for x in range(0, shape[0], patch_size[0]):
        for y in range(0, shape[1], patch_size[1]):
            for z in range(0, shape[2], patch_size[2]):
                idx = int(x/patch_size[0])
                idy = int(y/patch_size[1])
                idz = int(z/patch_size[2])
                index1D = to1D_index(idx, idy, idz, xMax, yMax)
                patch = numpy_image_padded[x:x + patch_size[0], y:y + patch_size[1], z:z + patch_size[2]]
                patches[index1D] = patch

    return patches


def to1D_index(x, y, z, xMax,yMax):
    return (z * xMax * yMax) + (y * xMax) + x


def to3D(idx, xMax, yMax):
    z = idx / (xMax * yMax)
    idx -= (z * xMax * yMax)
    y = idx / xMax
    x = idx % xMax
    return int(x), int(y), int(z)


def recreate_image_from_patches(original_image_size, list_patches, mode="extend"):
    patch_size = np.array(list_patches[0].shape)
    dimension_size = np.array(np.ceil(np.array(original_image_size)/patch_size), dtype=np.int64)

    size_image = np.array((patch_size*dimension_size), dtype=np.int64)
    xMax = dimension_size[0]
    yMax = dimension_size[1]

    image = np.zeros(size_image)

    for x in range(0, size_image[0], patch_size[0]):
        for y in range(0, size_image[1], patch_size[1]):
            for z in range(0, size_image[2], patch_size[2]):

                idx = int(x/patch_size[0])
                idy = int(y/patch_size[1])
                idz = int(z/patch_size[2])
                index1D = to1D_index(idx, idy, idz, xMax, yMax)

                image[x:x + patch_size[0], y:y + patch_size[1], z:z + patch_size[2]] = list_patches[index1D]

    if mode == "extend":
        ox, oy, oz = original_image_size
        image = image[:ox, :oy, :oz]


def create_and_save_patches(input_list, label_list, patching_save_path, patch_size=[32, 32, 32]):
    assert (len(input_list) == len(label_list))
    dir_patch_size = "x".join([str(x) for x in patch_size])
    patching_save_path = create_if_not_exists(os.path.join(patching_save_path, dir_patch_size))

    for subject in range(len(input_list)):
        x = input_list[subject]
        y = label_list[subject]
        input_patches = create_patches_from_images(x, patch_size)
        label_patches = create_patches_from_images(y, patch_size)
        for i in range(len(input_patches)):
            brain_scan = nb.Nifti1Image(input_patches[i], np.eye(4))
            lesion = nb.Nifti1Image(label_patches[i], np.eye(4))
            nb.save(brain_scan, os.path.join(patching_save_path, "/inputs/input_p{}-{}.nii".format(subject, i)))
            nb.save(lesion, os.path.join(patching_save_path, "/masks/label_p{}-{}.nii".format(subject, i)))


def transform_atlas_to_patches(altas_path, save_path, patch_size=[32, 32, 32]):
    sites = os.listdir(altas_path)
    create_if_not_exists(os.path.join(save_path, "inputs"))
    create_if_not_exists(os.path.join(save_path, "masks"))

    with progressbar.ProgressBar(max_value=len(sites)) as bar:
        percent = 0
        for site in sites:
            site_path = os.path.join(altas_path, site)
            patients = os.listdir(site_path)
            bar.update(percent)
            with progressbar.ProgressBar(max_value=len(sites)) as bar_patient:
                bar_patient.update(0)
                j=0
                for patient in patients:
                    j = j+1
                    # Read data for this patient
                    for t in os.listdir(os.path.join(site_path, patient)):
                        if not t == "t01" or patient == "031916":
                            # print("Found folder {} instead of T01 for patient {}".format(t, patient))
                            continue
                        brain_structure_path = os.path.join(site_path, patient, t, "output.nii")
                        lesion_path = os.path.join(site_path, patient, t, "{}_LesionSmooth_stx.nii".format(patient))
                        try:
                            brain_structure = nb.load(brain_structure_path).get_data()
                            lesion = nb.load(lesion_path).get_data()

                            # Create patches
                            x = brain_structure
                            y = lesion
                            input_patches = create_patches_from_images(x, patch_size)
                            label_patches = create_patches_from_images(y, patch_size)

                            # Save each patch
                            for i in range(len(input_patches)):
                                brain_scan = nb.Nifti1Image(input_patches[i], np.eye(4))
                                lesion = nb.Nifti1Image(label_patches[i], np.eye(4))
                                nb.save(brain_scan,
                                        os.path.join(save_path, "inputs", "input_{}-p{}-{}.nii".format(site, patient, i)))
                                nb.save(lesion,
                                        os.path.join(save_path, "masks", "label_{}-p{}-{}.nii".format(site, patient, i)))

                        except Exception as e:
                            print("Error reading patient {} ({})".format(patient, os.path.basename(site_path)))
                            print(e.with_traceback())
                            continue
                    bar_patient.update(j)


            percent = percent + 1

    return save_path


def save_set_to_folder(x_paths, y_paths, data_path, save_path):
    create_if_not_exists(save_path)
    with progressbar.ProgressBar(max_value=len(x_paths)) as bar:
        bar.update(0)
        old_folder_input = os.path.join(data_path, "inputs")
        old_folder_mask = os.path.join(data_path, "masks")
        new_folder_input = os.path.join(save_path, "inputs")
        new_folder_mask = os.path.join(save_path, "masks")
        create_if_not_exists(new_folder_input)
        create_if_not_exists(new_folder_mask)
        i = 0
        for x, y in zip(x_paths, y_paths):
            path_x = x.replace(old_folder_input, new_folder_input)
            path_y = y.replace(old_folder_mask, new_folder_mask)
            copyfile(x, path_x)
            copyfile(y, path_y)
            i = i + 1
            bar.update(i)


def splits_sets(data_path, save_path=None, ratios=[0.6,0.15,0.25], seed=42):
    assert(sum(ratios) == 1)
    if len(ratios) < 3:
        raise Exception("Ratio should be [train_ratio, val_ratio, test_ratio]")
    if save_path is None:
        save_path = data_path

    input_folder = os.path.join(data_path, "inputs")
    masks_folder = os.path.join(data_path, "masks")
    x = [os.path.join(input_folder,x) for x in os.listdir(input_folder)]
    y = [os.path.join(masks_folder,x) for x in os.listdir(masks_folder)]

    train_ratio = ratios[0]
    val_ratio = ratios[1]
    test_ratio = ratios[2]

    # Splitting train/test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, random_state=seed)

    # Splitting train/val
    ratio_split_val = val_ratio/(val_ratio + train_ratio)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=ratio_split_val, random_state=seed)

    save_set_to_folder(x_train, y_train, data_path, os.path.join(save_path, "train"))
    save_set_to_folder(x_val, y_val, data_path, os.path.join(save_path, "validation"))
    save_set_to_folder(x_test, y_test, data_path, os.path.join(save_path, "test"))
