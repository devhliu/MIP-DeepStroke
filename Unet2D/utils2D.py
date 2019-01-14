"""
Created on Wed Aug 29 15:12:49 2018
@author: Nabila Abraham
"""
import numpy as np
import os
from sklearn.model_selection import train_test_split
from shutil import copyfile
import numpy as np

def normalize_numpy(v, new_max=1.0, new_min=0.0):
    min_v = np.min(v)
    max_v = np.max(v)
    if min_v == max_v:
        if max_v > 0:
            return v / max_v
        return v

    v_new = (v-min_v)*(new_max-new_min)/(max_v-min_v) + new_min
    return v_new


def create_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def transorm_to_2d_usable_data(x_list_batch, y_list_batch=None, depth=4):
    new_x_batch = channel_first_to_channel_last(x_list_batch)
    if y_list_batch is None: # return only one list
        return new_x_batch

    # else, reverse channel as well
    new_y_batch = channel_first_to_channel_last(y_list_batch)

    # Create zoom masks
    gt_mask = []
    for i in range(depth):
        zoom = new_y_batch[:, ::np.power(2, i), ::np.power(2, i), :]
        gt_mask.insert(0, zoom)

    return new_x_batch, gt_mask


def channel_first_to_channel_last(x_list):
    x_list_batch = np.array(x_list)
    x_shape = x_list_batch.shape
    new_x_batch = np.zeros([x_shape[0], x_shape[2], x_shape[3], x_shape[1]])

    for c in range(x_list_batch.shape[1]):
        new_x_batch[:, :, :, c] = x_list_batch[:, c, :, :]

    return new_x_batch



def save_set_to_folder(x_paths, y_paths, data_path, save_path):
    create_if_not_exists(save_path)
    old_folder_input = os.path.join(data_path, "inputs")
    old_folder_mask = os.path.join(data_path, "masks")
    new_folder_input = os.path.join(save_path, "inputs")
    new_folder_mask = os.path.join(save_path, "masks")
    create_if_not_exists(new_folder_input)
    create_if_not_exists(new_folder_mask)
    for x, y in zip(x_paths, y_paths):
        path_x = x.replace(old_folder_input, new_folder_input)
        path_y = y.replace(old_folder_mask, new_folder_mask)
        copyfile(x, path_x)
        copyfile(y, path_y)


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


def check_preds(ypred, ytrue):
    smooth = 1
    pred = np.ndarray.flatten(np.clip(ypred, 0, 1))
    gt = np.ndarray.flatten(np.clip(ytrue, 0, 1))
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    return np.round((2 * intersection + smooth) / (union + smooth), decimals=5)


def confusion(y_true, y_pred):
    smooth = 1
    y_pred_pos = np.round(np.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = np.round(np.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = (np.sum(y_pos * y_pred_pos) + smooth) / (np.sum(y_pos) + smooth)
    tn = (np.sum(y_neg * y_pred_neg) + smooth) / (np.sum(y_neg) + smooth)
    return [tp, tn]


def auc(y_true, y_pred):
    smooth = 1
    y_pred_pos = np.round(np.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = np.round(np.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = np.sum(y_pos * y_pred_pos)
    tn = np.sum(y_neg * y_pred_neg)
    fp = np.sum(y_neg * y_pred_pos)
    fn = np.sum(y_pos * y_pred_neg)
    tpr = (tp + smooth) / (tp + fn + smooth)  # recall
    tnr = (tn + smooth) / (tn + fp + smooth)
    prec = (tp + smooth) / (tp + fp + smooth)  # precision


    return [tpr, tnr, prec]