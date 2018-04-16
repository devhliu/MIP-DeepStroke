import keras

from keras import Sequential
from keras.layers import Conv3D
from keras.callbacks import TensorBoard
from argparse import ArgumentParser
import subprocess
import os
import nibabel as nb
import numpy as np
import progressbar
from sklearn.model_selection import train_test_split
from PIL import Image

def load_data_atlas_from_site(site_path):
    list_x = []
    list_y = []
    patients = os.listdir(site_path)
    with progressbar.ProgressBar(max_value=len(patients)) as bar:
        i=0
        for patient in patients:
            bar.update(i)
            for t in os.listdir(os.path.join(site_path, patient)):
                if not t == "t01" or patient=="031916":
                    #print("Found folder {} instead of T01 for patient {}".format(t, patient))
                    continue
                brain_struct_path = os.path.join(site_path, patient, t, "output.nii")
                lesion_path = os.path.join(site_path, patient, t, "{}_LesionSmooth_stx.nii".format(patient))
                try:
                    brain_struct = nb.load(brain_struct_path).get_data()
                    lesion = nb.load(lesion_path).get_data()
                except Exception as e:
                    print("Error reading patient {} ({})".format(patient,os.path.basename(site_path)))
                    print(e.with_traceback())
                    continue

                list_x.append(brain_struct)
                list_y.append(lesion)
        i=i+1

    return list_x, list_y


def create_if_not_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    return dir


def create_patches(data_tuple_list, patch_size=[32, 32, 32, 3]):
    for brain, lesion in data_tuple_list:
        pad_width = [(x, x) for x in patch_size[0:3]]  # Symetric padding of one patch
        padded_image = brain.pad(pad_width, mode="constant", constant_values=0)


    return data_tuple_list


def train(model, training_data, validation_data, logdir=None):

    tensorboard_callback = None
    if logdir is not None:
        log_path = create_if_not_exists(os.path.join(logdir, "logs"))
        tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=0, batch_size=32, write_graph=True,
                                           write_grads=True, write_images=True, embeddings_freq=0,
                                           embeddings_layer_names=None, embeddings_metadata=None)
        # Start Tensorboard
        subprocess.run(["tensorboard --logdir={}".format(log_path)])

    # Save checkpoint each 5 epochs
    checkpoint_path = create_if_not_exists(os.path.join(logdir,"checkpoints"))

    keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=False,
                                    save_weights_only=False, mode='auto', period=10)

    # Parameters
    validation_steps = 5   # Number of steps per evaluation (number of to pass)
    steps_per_epoch = 20   # Number of batches to pass before going to next epoch
    shuffle = True         # Shuffle the data before creating a batch
    batch_size = 32

    train_x, train_y = zip(*training_data)

    # Train the model, iterating on the data in batches of 32 samples
    history = model.fit(x=train_x, y=train_y, validation_data=validation_data, steps_per_epoch=len(train_x)/batch_size,
                        validation_steps = validation_steps, shuffle=shuffle, epochs=10000, batch_size=batch_size,
                        callbacks=[tensorboard_callback])

    return model, history


def to_patches(toPatch, PATCH_Z, STRIDE_PATCH_Z, PATCH_X, STRIDE_PATCH_X, PATCH_Y, STRIDE_PATCH_Y):

    shape = np.shape(toPatch)

    slices = shape[0]
    x_shape = shape[1]
    y_shape = shape[2]

    def check(x,y,z): #TODO: see why it doesnt work
        assert(((float)((x-PATCH_X)/STRIDE_PATCH_X)).is_integer())
        assert(((float)((y-PATCH_Y)/STRIDE_PATCH_Y)).is_integer())
        assert(((float)((z-PATCH_Z)/STRIDE_PATCH_Z)).is_integer())
    check(x_shape,y_shape,slices)

    patches = []
    for k in range(0, slices-PATCH_Z+STRIDE_PATCH_Z,STRIDE_PATCH_Z):
        for i in range(0,x_shape-PATCH_X+STRIDE_PATCH_X,STRIDE_PATCH_X):
            for j in range(0,y_shape-PATCH_Y+STRIDE_PATCH_Y,STRIDE_PATCH_Y):
                patches.append(toPatch[k:k+PATCH_Z,i:i+PATCH_X, j:j+PATCH_Y])
    print(np.shape(patches))
    return patches


def create_patches_from_images(numpy_image, patch_size, mode="extend"):

    patch_size_x = patch_size[0]
    patch_size_y = patch_size[1]
    patch_size_z = patch_size[2]


    shape = numpy_image.shape
    missing = np.array([patch_size[i]-(shape[i]%patch_size[i]) for i in range(len(patch_size))])
    numpy_image_padded = np.zeros(numpy_image.shape+missing)
    print(numpy_image_padded.shape)
    if mode is "extend":
            numpy_image_padded[:,:,:] = np.pad(numpy_image[:,:,:], [(0, missing[0]), (0, missing[1]), (0, missing[2])], mode="constant",
                                    constant_values=0)

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

                image[x:x + patch_size[0], y:y + patch_size[1], z:z + patch_size[2]] = patches[index1D]

    if mode == "extend":
        ox, oy, oz = original_image_size
        image = image[:ox, :oy, :oz]

    return image


def create_and_save_patches(input_list, label_list, patching_save_path, patch_size=[32, 32, 32]):
    assert(len(input_list)==len(label_list))
    dir_patch_size = "x".join(patch_size)
    patching_save_path = create_if_not_exists(os.path.join(patching_save_path, dir_patch_size))

    for subject in range(len(input_list)):
        x = input_list[subject]
        y = label_list[subject]
        input_patches = create_patches_from_images(x, patch_size)
        label_patches = create_patches_from_images(y, patch_size)
        for i in range(len(input_patches)):
            im = Image.fromarray(input_patches[i])
            im.save(os.path.join(patching_save_path, "input_p{}-{}.jpeg".format(subject, i)))
            im = Image.fromarray(label_patches[i])
            im.save(os.path.join(patching_save_path, "label_p{}-{}.jpeg".format(subject, i)))


if __name__ == '__main__':
    parser = ArgumentParser(description="Train a 3D Model unet")
    parser.add_argument("-l", "--logdir", help="Directory where to log the data for tensorboard",
                        default ="/home/snarduzz/")

    parser.add_argument("-d", "--datapath", help="Path to data folder (ATLAS)", default="/home/simon/Datasets/Stroke_DeepLearning_ATLASdataset")
    args = parser.parse_args()

    sites = [os.path.join(dataos.listdir(args.datapath)
    patch_size = [32, 32, 32]
    print("Saving patches...")
    with progressbar.ProgressBar(max_value=len(sites)) as bar:
        i=0
        for site in sites:
            x, y = load_data_atlas_from_site(site)
            save_dir = create_if_not_exists("Data/{}".format(site))
            create_and_save_patches(x, y, save_dir)
            i=i+1
            bar.update(i)
    print("Done!")