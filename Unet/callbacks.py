# Extends Keras' TensorBoard callback to include the Precision-Recall summary plugin.

import os

from keras.callbacks import TensorBoard, EarlyStopping

from tensorboard.plugins.pr_curve import summary as pr_summary
from tensorboard.plugins.scalar import summary as sc_summary
from tensorboard.plugins.image import summary as im_summary
from image_processing import create_patches_from_images
from predict import predict
import numpy as np
import tensorflow as tf
from scipy.misc import toimage
from io import StringIO,BytesIO
from PIL import Image
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
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc, 4)), str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

class TrainValTensorBoard(TensorBoard):
    def __init__(self, image=None, lesion=None, patch_size=None, layer=None, verbose=0, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

        # Get PR Curve
        self.pr_curve = kwargs.pop('pr_curve', True)
        self.initialized = False

        # Image
        if image is not None:
            if lesion is None or patch_size is None or layer is None:
                raise Exception("Please provide the following : \'image\',\'lesion\',\'patch_size\',\'layer\'")
            self.image = image
            self.lesion = lesion
            self.lesion_patches = create_patches_from_images(image, patch_size)
            self.image_patches = create_patches_from_images(lesion, patch_size)
            self.layer = layer
            self.patch_size = patch_size
            self.verbose = verbose

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
            self.val_writer.add_summary(summary, epoch, )


        # Add PR Curve
        self.__add_pr_curve(epoch)
        # add image
        self.__add_image(epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)


    def log_images(self, tag, images, step):
        """Logs a list of images."""

        image_summaries = []
        for image_num, image in enumerate(images):
            # Write the image to a string
            try:
                # Python 2.7
                s = StringIO()
                Image.fromarray(image).save(s, format="png")
            except TypeError:
                # Python 3.X
                s = BytesIO()
                Image.fromarray(image).save(s, format="png")
            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=image.shape[0],
                                       width=image.shape[1])
            # Create a Summary value
            image_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, image_num),
                                                    image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=image_summaries)
        self.val_writer.add_summary(summary, step)

    def __add_image(self, epoch):
        if self.lesion_patches and self.image_patches:
            # Predict the output.
            predicted_image = predict(self.image, self.model, self.patch_size, verbose=self.verbose)

            pred_image = predicted_image[:, :, self.layer]
            image_original = self.image[:, :, self.layer]
            lesion_original = self.lesion[:, :, self.layer]

            # RGB
            merged_image = np.zeros([pred_image.shape[0], pred_image.shape[1], 3])
            merged_image[:, :, 0] = pred_image
            merged_image[:, :, 1] = image_original
            merged_image[:, :, 2] = lesion_original

            #pred_tensor = tf.convert_to_tensor(pred_image.reshape(1, pred_image.shape[0], pred_image.shape[1], 1))
            #image_tensor = image_original.reshape(1, image_original.shape[0], image_original.shape[1], 1)
            #lesion_tensor = lesion_original.reshape(1, lesion_original.shape[0], lesion_original.shape[1], 1)

            self.log_images(tag="prediction", images=[merged_image], step=epoch)
            """
            tensor_images = [pred_tensor, image_tensor, lesion_tensor, merged_image]
            #pred_summary = tf.summary.image(name="prediction", tensor=pred_tensor, max_outputs=1)
            tf.summary.image(name="prediction", tensor=pred_tensor, max_outputs=1)
            img_summary = tf.Summary()
            summary_value = img_summary.value.add()
            summary_value =


            #image_summary = tf.summary.image("input", image_tensor, max_outputs=10)
            #lesion_summay = tf.summary.image("lesion", lesion_tensor, max_outputs=10)
            #merged_summary = tf.summary.image("merged", merged_image, max_outputs=10)

            # Run and add summary.
            summary = tf.summary.merge_all()
            result = self.sess.run([summary])
            self.val_writer.add_summary(result[0], epoch)
            #self.val_writer.add_summary(result[1])
            #self.val_writer.add_summary(result[2])
            #self.val_writer.add_summary(result[3])
            """

    def __add_pr_curve(self, epoch):
        if self.pr_curve and self.validation_data:
            print("INSIDE PR CURVE")
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
            self.val_writer.add_summary(result[1], epoch)
            self.val_writer.add_summary(result[2], epoch)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
