from functools import partial

from keras import backend as K
import tensorflow as tf
import numpy as np
from sklearn.metrics import auc, roc_curve


def auc_score(y_true, y_pred):
    # any tensorflow metric
    auc, update_op = tf.metrics.auc(y_true, y_pred, curve="PR", name="auc")
    return auc


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def precision(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    TP, _ = tf.metrics.true_positives(y_true_f, y_pred_f)
    FP, _ = tf.metrics.false_positives(y_true_f, y_pred_f)
    return K.division(TP, (K.sum(TP, FP)))


def recall(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    r = tf.metrics.recall(y_true_f,y_pred_f)
    return K.eval(r)


def PR(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    thresholds = np.arange(0, 1, 0.05)
    r = tf.metrics.recall_at_thresholds(y_true_f, y_pred_f, thresholds)
    p = tf.metrics.precision_at_thresholds(y_true_f, y_pred_f, thresholds)
    return K.division(p/r)


def AUC(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return K.eval(tf.metrics.auc(y_true_f, y_pred_f))


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
