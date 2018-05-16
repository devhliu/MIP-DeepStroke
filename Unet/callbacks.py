# Extends Keras' TensorBoard callback to include the Precision-Recall summary plugin.

import os

from keras.callbacks import TensorBoard, EarlyStopping

from tensorboard.plugins.pr_curve import summary as pr_summary
from tensorboard.plugins.scalar import summary as sc_summary
from image_processing import create_patches_from_images
from predict import predict
import numpy as np
import tensorflow as tf

from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback


class roc_callback(Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


class ImageTensorBoard(TensorBoard):
    def __init__(self, image, lesion, patch_size, layer, verbose=0, *args, **kwargs):
        super(ImageTensorBoard, self).__init__(*args, **kwargs)
        self.image = image
        self.lesion = lesion
        self.lesion_patches = create_patches_from_images(image, patch_size)
        self.image_patches = create_patches_from_images(lesion, patch_size)
        self.layer = layer
        self.patch_size = patch_size
        self.verbose = verbose

    def set_model(self, model):
        super(ImageTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        super(ImageTensorBoard, self).on_epoch_end(epoch, logs)

        if self.lesion_patches and self.image_patches:
            # Predict the output.
            predicted_image = predict(self.image, self.model, self.patch_size, verbose=self.verbose)

            pred_image = predicted_image[:, :, self.layer]
            image_original = self.image[:, :, self.layer]
            lesion_original = self.lesion[:, :, self.layer]

            # RGB
            merged_image = np.array(pred_image.shape, 3)
            merged_image[:, : , 0] = pred_image
            merged_image[:, :, 1] = image_original
            merged_image[:, :, 2] = lesion_original

            pred_summary = tf.summary.image("prediction", pred_image)
            image_summary = tf.summary.image("input", image_original)
            lesion_summay = tf.summary.image("lesion", lesion_original)
            merged_summary = tf.summary.image("merged", merged_image)

            # Run and add summary.
            result = self.sess.run([pred_summary, image_summary, lesion_summay, merged_summary])
            self.writer.add_summary(result[0], epoch)
            self.writer.add_summary(result[1], epoch)
            self.writer.add_summary(result[2], epoch)
            self.writer.add_summary(result[3], epoch)
        self.writer.flush()


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

        # Get PR Curve
        self.pr_curve = kwargs.pop('pr_curve', True)
        self.initialized = False

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

        if self.pr_curve:
            # Get the prediction and label tensor placeholders.
            predictions = self.model._feed_outputs[0]
            labels = tf.cast(self.model._feed_targets[0], tf.bool)
            # Create the PR summary OP.
            self.pr_summary = pr_summary.op(name="pr_curve",
                                            predictions=predictions,
                                            labels=labels,
                                            display_name='Precision-Recall Curve')

            self.auc, ops = tf.metrics.auc(labels, predictions, curve="PR", name="auc_metric")
            self.auc_summary = sc_summary.op(name="auc", data=self.auc, description="Area Under Curve")


    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

        # Add PR Curve
        if self.pr_curve and self.validation_data:
            # Get the tensors again.
            tensors = self.model._feed_targets + self.model._feed_outputs
            # Predict the output.
            predictions = self.model.predict(self.validation_data[:-2])
            # Build the dictionary mapping the tensor to the data.
            val_data = [self.validation_data[-2], predictions]
            feed_dict = dict(zip(tensors, val_data))
            # Run and add summary.
            if not self.initialized:
                self.sess.run(tf.local_variables_initializer())
                self.initialized = True

            result = self.sess.run([self.pr_summary, self.auc, self.auc_summary], feed_dict=feed_dict)
            self.val_writer.add_summary(result[0], epoch)
        self.val_writer.flush()

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
