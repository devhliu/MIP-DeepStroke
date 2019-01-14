import numpy as np
import keras
import os
import nibabel as nb
from image_augmentation import randomly_augment


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_directory, folders_input, folders_output, batch_size, skip_blank=False, logfile=None,
                 shuffle=True, augment_prob=None, depth=1):
        'Initialization'
        if augment_prob is None:
            augment_prob = {"rotation": 0.0,
                            "rotxmax": 90.0,
                            "rotymax": 90.0,
                            "rotzmax": 90.0,
                            "rotation_step": 1.0,
                            "salt_and_pepper": 0.0,
                            "flip": 0.0,
                            "contrast_and_brightness": 0.0,
                            "only_positives": False}

        self.data_directory = data_directory
        self.folders_input = folders_input
        self.folders_output = folders_output
        self.batch_size = batch_size
        self.skip_blank = skip_blank
        self.logfile = logfile
        self.shuffle = shuffle
        self.augment_prob = augment_prob
        self.n_channels_input = len(folders_input)
        self.n_channels_output = len(folders_output)

        # list files in folder
        self.image_paths = os.listdir(os.path.join(data_directory, folders_input[0]))
        self.list_IDs = np.arange(len(self.image_paths))

        example_image = os.path.join(self.data_directory, self.folders_input[0], self.image_paths[0])
        self.dim = nb.load(example_image).get_data().shape
        self.actual_index = -1  # will be incremented in next()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def next(self):
        self.actual_index = self.actual_index+1
        if self.actual_index > self.__len__()-1:
            self.actual_index = 0
        return self.__getitem__(self.actual_index)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # Channel first
        X = np.empty((self.batch_size, self.n_channels_input, *self.dim))
        y = np.empty((self.batch_size, self.n_channels_output, *self.dim), dtype=int)

        if self.logfile:
            with open(self.logfile, "a") as f:
                f.write("--- {} ---\n".format(list_IDs_temp))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # load image
            image_path = self.image_paths[ID]

            # for log
            inputs_paths = []
            outputs_paths = []

            patch_by_channels = np.empty((self.n_channels_input, *self.dim))
            for channel in range(self.n_channels_input):
                #load corresponding channel image
                image_path_channel = image_path.replace(self.folders_input[0], self.folders_input[channel])
                patch_channel = nb.load(os.path.join(self.data_directory, self.folders_input[channel], image_path_channel)).get_data()
                patch_by_channels[channel, ...] = patch_channel
                #log
                inputs_paths.append(image_path_channel)

            labels_by_channels = np.empty((self.n_channels_output, *self.dim))
            for channel in range(self.n_channels_output):
                #load corresponding channel image
                image_path_channel = image_path.replace(self.folders_input[0], self.folders_output[channel])
                label_channel = nb.load(os.path.join(self.data_directory, self.folders_output[channel], image_path_channel)).get_data()
                labels_by_channels[channel, ...]= label_channel
                #log
                outputs_paths.append(image_path_channel)

            patch_by_channels, labels_by_channels = self.__data_augmentation(patch_by_channels, labels_by_channels)
            # Store sample
            X[i, ...] = patch_by_channels
            # Store class
            y[i, ...] = labels_by_channels

            if self.logfile:
                with open(self.logfile, "a") as f:
                    paths = inputs_paths + [" <--> "] +outputs_paths
                    f.write("{} - {}\n".format(i, paths))

        return X, y

    def __data_augmentation(self, inputs, targets):
        augment_prob = self.augment_prob
        if augment_prob["only_positives"] and np.sum(targets) > 0:
            inputs, targets = randomly_augment(inputs, targets, prob=augment_prob)
        else:
            inputs, targets = randomly_augment(inputs, targets, prob=augment_prob)

        return inputs, targets
