import os
import nibabel as nb
import numpy as np
import progressbar
from utils import create_if_not_exists, normalize_numpy


def preprocess_image(img, preprocessing="standardize"):
    if preprocessing is "normalize":
        return normalize_numpy(img)
    if preprocessing is "standardize":
        return standardize(img)

def standardize(img):
    mean = np.mean(img)
    std = np.std(img)
    img_std = (img-mean)/std

    #Clipping values between -2 and 3 to remove outliers
    img_clipped = np.clip(img_std, -2, 3)

    # scaling between -1 and 1
    img_scaled = normalize_numpy(img_clipped, new_min=0, new_max=1)
    return img_scaled

def load_data_atlas_for_patient(patient_path):
    patient = os.path.basename(patient_path)
    #list_x = []
    #list_y = []
    for t in os.listdir(patient_path):
        if not t == "t01" or os.path.basename(patient_path) == "031916":
            # print("Found folder {} instead of T01 for patient {}".format(t, patient))
            continue
        brain_structure_path = os.path.join(patient_path, t, "output.nii")
        lesion_path = os.path.join(patient_path, t, "{}_LesionSmooth_stx.nii".format(patient))
        try:
            brain_structure = nb.load(brain_structure_path).get_data()
            lesion = nb.load(lesion_path).get_data()
        except Exception as e:
            print("Error reading patient {} ({})".format(patient_path))
            print(e.with_traceback())
            continue

        return brain_structure, lesion

    #    list_x.append(brain_structure)
    #    list_y.append(lesion)
    #return list_x, list_y


def load_data_atlas_from_site(site_path):
    list_x = []
    list_y = []
    patients = os.listdir(site_path)
    with progressbar.ProgressBar(max_value=len(patients)) as bar:
        i = 0
        for patient in patients:
            bar.update(i)
            brain_structure, lesion = load_data_atlas_for_patient(os.path.join(site_path,patient))

            list_x.append(brain_structure)
            list_y.append(lesion)
            i = i + 1

    return list_x, list_y


def create_patches_from_images(numpy_image, patch_size, mode="extend", augment=False, patch_divider=4):
    if (patch_size[0] == 1 or patch_size[1] == 1 or patch_size[2] == 1):
        raise Exception("Patch size should at least be >[2,2,2] ")
    shape = numpy_image.shape
    missing = np.array([patch_size[i] - (shape[i] % patch_size[i]) for i in range(len(patch_size))])
    numpy_image_padded = np.zeros(numpy_image.shape + missing)

    if mode is "extend":
        numpy_image_padded[:, :, :] = np.pad(numpy_image[:, :, :], [(0, missing[0]), (0, missing[1]), (0, missing[2])],
                                             mode="constant", constant_values=0)

    shape = numpy_image_padded.shape
    dimension_size = np.array(np.ceil(np.array(numpy_image.shape) / patch_size), dtype=np.int64)
    patches = []
    if augment:
        # Create dataset using strides
        PATCH_X = patch_size[0]
        PATCH_Y = patch_size[1]
        PATCH_Z = patch_size[2]
        STRIDE_PATCH_X = int(patch_size[0] / patch_divider)
        STRIDE_PATCH_Y = int(patch_size[1] / patch_divider)
        STRIDE_PATCH_Z = int(patch_size[2] / patch_divider)
        patches = to_patches_3d(numpy_image_padded, PATCH_X, STRIDE_PATCH_X, PATCH_Y, STRIDE_PATCH_Y, PATCH_Z,
                                STRIDE_PATCH_Z)
    else:
        for x in range(0, shape[0], patch_size[0]):
            for y in range(0, shape[1], patch_size[1]):
                for z in range(0, shape[2], patch_size[2]):
                    patch = numpy_image_padded[x:x + patch_size[0], y:y + patch_size[1], z:z + patch_size[2]]
                    patches.append(patch)

    return patches


def create_extra_patches(numpy_image, numpy_lesion, patch_size, limit=0, sampled_indices=None):
    """
    Create patches with some lesion centered in the middle
    :param numpy_image: numpy array representing the original mri file
    :param numpy_lesion: numpy array representing the original lesion file
    :param patch_size: 3D tuple for the size of one patch
    :param limit: number of patches to return
    :return: two list of numpy arrays, one for the images and one for the corresponding lesion
    """
    image_padded = np.pad(numpy_image, [(int(patch_size[0] / 2), int(patch_size[0] / 2)),
                                  (int(patch_size[1] / 2), int(patch_size[1] / 2)),
                                  (int(patch_size[2] / 2), int(patch_size[2] / 2))],
                          mode="constant", constant_values=0)

    lesion_padded = np.pad(numpy_lesion, [(int(patch_size[0] / 2), int(patch_size[0] / 2)),
                                    (int(patch_size[1] / 2), int(patch_size[1] / 2)),
                                    (int(patch_size[2] / 2), int(patch_size[2] / 2))],
                           mode="constant", constant_values=0)

    pos_indices = np.array(np.where(lesion_padded >= 1)).T

    patches_image = []
    patches_lesion = []

    random_indices = np.arange(len(pos_indices))
    np.random.shuffle(random_indices)
    if not sampled_indices:
        sampled_indices = [pos_indices[x] for x in random_indices]
    if limit > 0:
        sampled_indices = sampled_indices[:limit]

    for x, y, z in sampled_indices:
            patch_image = create_patch_around(x, y, z, patch_size, image_padded)
            patch_lesion = create_patch_around(x, y, z, patch_size, lesion_padded)

            patches_image.append(patch_image)
            patches_lesion.append(patch_lesion)

    return patches_image, patches_lesion


def create_extra_patches_from_list(images, lesion, patch_size, limit=0):

    # Pad the lesion to get squared image, should correspond to pad in create_extra_patches
    lesion_padded = np.pad(lesion, [(int(patch_size[0] / 2), int(patch_size[0] / 2)),
                                          (int(patch_size[1] / 2), int(patch_size[1] / 2)),
                                          (int(patch_size[2] / 2), int(patch_size[2] / 2))],
                           mode="constant", constant_values=0)

    # Compute indices of "centered lesion"
    pos_indices = np.array(np.where(lesion_padded >= 1)).T
    random_indices = np.arange(len(pos_indices))
    np.random.shuffle(random_indices)
    sampled_indices = [pos_indices[x] for x in random_indices]

    list_extra_patches = []
    patches_lesion = None
    for im in images:
        # From indices, create extra patches for every kind of image
        patches_image, patches_lesion = create_extra_patches(im, lesion, patch_size, limit, sampled_indices)
        list_extra_patches.append(patches_image)

    # Last array is the one containing extra patches for the lesion
    list_extra_patches.append(patches_lesion)
    return list_extra_patches


# Create patches from a scan - toPatch should be (x,y,z) - the output of the function is (nb_patches,x,y,z)
def to_patches_3d(toPatch,PATCH_X,STRIDE_PATCH_X,PATCH_Y,STRIDE_PATCH_Y,PATCH_Z,STRIDE_PATCH_Z):

    shape = toPatch.shape
    x_shape = shape[0]
    y_shape = shape[1]
    slices = shape[2]

    def check(x,y,z):
        try:
            val = (float)((x-PATCH_X)/STRIDE_PATCH_X)
            assert((val).is_integer())
        except:
            print("{}-{}/{} = {} is not integer.".format(x,PATCH_X,STRIDE_PATCH_X, val))

        try:
            val = (float)((y-PATCH_Y)/STRIDE_PATCH_Y)
            assert((val).is_integer())
        except:
            print("{}-{}/{} = {} is not integer.".format(y,PATCH_Y,STRIDE_PATCH_Y, val))

        try:
            val = (float)((z - PATCH_Z) / STRIDE_PATCH_Z)
            assert ((val).is_integer())
        except:
            print("{}-{}/{} = {} is not integer.".format(z, PATCH_Z, STRIDE_PATCH_Z, val))

    check(x_shape,y_shape,slices)

    #patches = np.empty((0,PATCH_X,PATCH_Y,PATCH_Z))
    patches = []
    for k in range(0,slices-PATCH_Z+STRIDE_PATCH_Z,STRIDE_PATCH_Z):
        for i in range(0,x_shape-PATCH_X+STRIDE_PATCH_X,STRIDE_PATCH_X):
            for j in range(0,y_shape-PATCH_Y+STRIDE_PATCH_Y,STRIDE_PATCH_Y):
                #cut = np.reshape(toPatch[i:i+PATCH_X, j:j+PATCH_Y,k:k+PATCH_Z],(1,PATCH_X,PATCH_Y,PATCH_Z))
                # patches = np.append(patches,cut,axis=0)
                cut = toPatch[i:i+PATCH_X, j:j+PATCH_Y,k:k+PATCH_Z]
                patches.append(cut)

    return patches


# Augment the patches with new patches where the center pixel is part of a lesion - for 3D
def to_patches_3d_augmented_with_1(mask,structural,PATCH_X,STRIDE_PATCH_X,PATCH_Y,STRIDE_PATCH_Y,PATCH_Z,STRIDE_PATCH_Z):

    shape = mask.shape
    x_shape = shape[0]
    y_shape = shape[1]
    slices = shape[2]

    def check(x,y,z):
        assert(((float)((x-PATCH_X)/STRIDE_PATCH_X)).is_integer())
        assert(((float)((y-PATCH_Y)/STRIDE_PATCH_Y)).is_integer())
        assert(((float)((z-PATCH_Z)/STRIDE_PATCH_Z)).is_integer())

    check(x_shape,y_shape,slices)


    patches_mask = np.empty((0,PATCH_X,PATCH_Y,PATCH_Z))
    patches_struc = np.empty((0,PATCH_X,PATCH_Y,PATCH_Z))

    mid_x = PATCH_X/2
    mid_y = PATCH_Y/2
    mid_z = PATCH_Z/2

    for k in range(mid_z,slices-mid_z):
        for i in range(mid_x,x_shape-mid_x):
            for j in range(mid_y,y_shape-mid_y):
                if(mask[i][j][k]==1):
                    cut = mask[i-mid_x:i+mid_x,j-mid_y:j+mid_y,k-mid_z:k+mid_z]
                    patches_mask = np.append(patches_mask,cut,axis=0)
                    cut = structural[i-mid_x:i+mid_x,j-mid_y:j+mid_y,k-mid_z:k+mid_z]
                    patches_struc = np.append(patches_struc,cut,axis=0)

    return patches_mask,patches_struc



def create_patch_around(center_x, center_y, center_z, patch_size, image):
    x_begin = center_x - int(patch_size[0] / 2.0)
    x_end = center_x + int(patch_size[0] / 2.0)
    y_begin = center_y - int(patch_size[1] / 2.0)
    y_end = center_y + int(patch_size[1] / 2.0)
    z_begin = center_z - int(patch_size[2] / 2.0)
    z_end = center_z + int(patch_size[2] / 2.0)

    patch = image[x_begin:x_end, y_begin:y_end, z_begin:z_end]
    return patch


def get_lesion_ratio(patches_list):
    """Compute the ratio of lesioned tissue in total list of patches"""
    ratio = 0
    for p in patches_list:
        s = np.array(p.shape)/2
        s = s.astype(np.int64)
        center = p[s[0], s[1], s[2]]
        if(center>0):
            ratio+=1

    return float(ratio)/float(len(patches_list))


def to1D_index(x, y, z, xMax, yMax, zMax):
    k =  (z * xMax * yMax) + (y * xMax) + x
    return k


def to3D(idx, xMax, yMax):
    z = idx / (xMax * yMax)
    idx -= (z * xMax * yMax)
    y = idx / xMax
    x = idx % xMax
    return int(x), int(y), int(z)


def recreate_image_from_patches(original_image_size, list_patches, mode="extend"):
    patch_size = np.array(list_patches[0].shape)
    dimension_size = np.array(np.ceil(np.array(original_image_size) / patch_size), dtype=np.int64)

    if type(list_patches) is np.ndarray:
        list_patches = list_patches.tolist()

    size_image = np.array((patch_size * dimension_size), dtype=np.int64)
    image = np.zeros(size_image)

    for x in range(0, size_image[0] - 1, patch_size[0]):
        for y in range(0, size_image[1] - 1, patch_size[1]):
            for z in range(0, size_image[2] - 1, patch_size[2]):
                p = list_patches.pop(0)
                image[x:x + patch_size[0], y:y + patch_size[1], z:z + patch_size[2]] = p

    if mode == "extend":
        ox, oy, oz = original_image_size
        image = image[:ox, :oy, :oz]

    return image


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
            with progressbar.ProgressBar(max_value=len(patients)) as bar_patient:
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
