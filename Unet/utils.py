import os
import progressbar
from sklearn.model_selection import train_test_split
from shutil import copyfile


def normalize_numpy(v, max_value=1.0, min_value=0.0):
    return v*max_value/v.max()

def create_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


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


def splits_sets(data_path, save_path=None, ratios=[0.7, 0.2, 0.1], seed=None):
    assert(sum(ratios) == 1)
    if len(ratios) < 3:
        raise Exception("Ratio should be [train_ratio, val_ratio, test_ratio]")
    if save_path is None:
        save_path = data_path

    input_folder = os.path.join(data_path, "inputs")
    masks_folder = os.path.join(data_path, "masks")
    x = [os.path.join(input_folder, x) for x in os.listdir(input_folder)]
    y = [os.path.join(masks_folder, x) for x in os.listdir(masks_folder)]

    x_train, x_test, x_val, y_train, y_test, y_val = split_train_test_val(x, y, ratios, seed)

    save_set_to_folder(x_train, y_train, data_path, os.path.join(save_path, "train"))
    save_set_to_folder(x_val, y_val, data_path, os.path.join(save_path, "validation"))
    save_set_to_folder(x_test, y_test, data_path, os.path.join(save_path, "test"))


def split_train_test_val(x, y, ratios=[0.7, 0.2, 0.1], seed=None):
    """
    Splits the inputs and labels in three sets : train,test and validation.
    :param x: list of inputs
    :param y: list of labels
    :param ratios: 3-tuple containing the ratios [train, test, val]
    :param seed: a seed used in case of shuffle
    :return: 3 sets : train, test and validation in the form of 6-tuple (x_train, x_test, x_val, y_train, y_test, y_val)
    """
    train_ratio = ratios[0]
    test_ratio = ratios[1]
    val_ratio = ratios[2]

    # Splitting train/test
    if seed is not None:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, random_state=seed)
    else:
        train_val_ratio = train_ratio + val_ratio
        data = list(zip(x, y))
        num_train = int(train_val_ratio * len(data))
        x_train, y_train = zip(*data[:num_train])
        x_test, y_test = zip(*data[num_train:])

    # Splitting train/val
    ratio_split_val = val_ratio / (val_ratio + train_ratio)
    if seed is not None:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=ratio_split_val,
                                                          random_state=seed)
    else:
        data = list(zip(x_train, y_train))
        num_train = 1 - int(ratio_split_val * len(data))  # 1-validation_ratio
        x_train, y_train = zip(*data[:num_train])
        x_val, y_val = zip(*data[num_train:])

    return x_train, x_test, x_val, y_train, y_test, y_val
