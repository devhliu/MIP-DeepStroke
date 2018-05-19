# Extends Keras' TensorBoard callback to include the Precision-Recall summary plugin.

import os

from keras.callbacks import TensorBoard, EarlyStopping

from tensorboard.plugins.pr_curve import summary as pr_summary
from tensorboard.plugins.scalar import summary as sc_summary
from tensorboard.plugins.image import summary as im_summary
from image_processing import create_patches_from_images
from utils import normalize_numpy
from predict import predict
import numpy as np
import tensorflow as tf
import skimage.io
from io import StringIO,BytesIO
from PIL import Image
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm


class TrainValTensorBoard(TensorBoard):
    def __init__(self, image=None, lesion=None, patch_size=None, layer=None,  training_generator=None, validation_generator=None, validation_steps=None,
                 verbose=0, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

        # Get PR Curve
        self.pr_curve = kwargs.pop('pr_curve', True)
        self.initialized = False
        self.validation_generator = validation_generator
        self.validation_steps = validation_steps
        self.training_generator = training_generator
        self.lesion_patches = None
        self.image_patches = None

        # Image
        if image is not None:
            if lesion is None or patch_size is None or layer is None:
                raise Exception("Please provide the following : \'image\',\'lesion\',\'patch_size\',\'layer\'")
            self.image = normalize_numpy(image)
            self.lesion = normalize_numpy(lesion)
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
        self.__add_batch_visualization(epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)


    def log_images(self, tag, images, step, writer=None):
        """Logs a list of images."""
        if writer is None:
            writer = self.val_writer

        image_summaries = []
        for image_num, image in enumerate(images):
            # Write the image to a string
            try:
                # Python 2.7
                s = StringIO()
                skimage.io.imsave(s, image)
            except TypeError:
                # Python 3.X
                s = BytesIO()
                skimage.io.imsave(s, image)
            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=image.shape[0],
                                       width=image.shape[1])
            # Create a Summary value
            image_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, image_num),
                                                    image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=image_summaries)
        writer.add_summary(summary, step)

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

            self.log_images(tag="prediction", images=[pred_image, lesion_original, image_original, merged_image], step=epoch)


    def __merge_images(self, images):
        # Create merged image
        nb_images = len(images)
        square = np.ceil(np.sqrt(nb_images))
        im_shape = images[0].shape
        while len(images) != square:
            # add blank images to obtain a square
            blank_image = np.zeros(im_shape)
            blank_image[:, :, 1] = 1
            images.append(blank_image)

        merged_image = np.zeros(square*im_shape[0], square*im_shape[1], im_shape[2])
        for i in range(0, square):
            for j in range(0, square):
                idx = i*im_shape[0]
                idy = j*im_shape[1]
                merged_image[idx:idx+im_shape[0], idy:idy+im_shape[1], :] = images[i*square+j]

        return merged_image


    def __add_batch_visualization(self, epoch):
        batch = next(self.training_generator)
        b = next(self.validation_generator)
        images = []
        for x, y in zip(batch[0], batch[1]):
            image = x[0,:,:,:]
            lesion = y[0,:,:,:]
            layer = int(image.shape[2] / 2)
            image_layer = image[:, :, layer]
            lesion_layer = lesion[:, :, layer]
            merged_image = np.zeros([image_layer.shape[0], image_layer.shape[1], 3])
            merged_image[:, :, 0] = lesion_layer
            merged_image[:, :, 1] = 0
            merged_image[:, :, 2] = image_layer
            images.append(merged_image)

        image_merged = self.__merge_images(images)

        self.log_images(tag="Batch", images=[image_merged], step=epoch)



    def __add_batch_viz(self, tag, epoch):
        images = []
        if tag is "training_batch":
            generator = self.training_generator
            writer = self.writer
        else:
            generator = self.validation_generator
            writer = self.val_writer
        if generator:
            for batch in next(generator):
                images_x = batch[0]
                lesions_y = batch[1]
                for image, lesion in zip(images_x,lesions_y):
                    layer = int(image.shape[2]/2)
                    image_layer = image[:, :, layer]
                    lesion_layer = lesion[:, :, layer]
                    merged_image = np.zeros([image_layer.shape[0], image_layer.shape[1], 3])
                    merged_image[:, :, 0] = lesion_layer
                    merged_image[:, :, 2] = image_layer
                    images.append(merged_image)



                self.log_images(tag=tag, images=images, step=epoch, writer=writer)

    def __add_pr_curve(self, epoch):
        if self.pr_curve and self.validation_generator:
            if self.validation_steps is None:
                raise Exception("Please provide validation_step argument")
            # Get the tensors again.
            tensors = self.model._feed_targets + self.model._feed_outputs
            # Predict the output.
            mean_roc = []
            mean_precision = []
            mean_recall = []

            generator = self.validation_generator
            for b in tqdm(range(self.validation_steps)):
                x,y = next(generator)

                pred_batch = self.model.predict_on_batch(y)

                if len(np.unique(y)) > 1:
                    roc = roc_auc_score(y.flatten(), pred_batch.flatten())
                    mean_roc.append(roc)

                    #precision, recall, _ = precision_recall_curve(y.flatten(), pred_batch.flatten())
                    #mean_precision.append(precision)
                    #mean_recall.append(recall)

            #mean_recall = np.mean(mean_recall, axis=0)
            #mean_precision = np.mean(mean_precision, axis=0)
            if len(mean_roc) == 0:
                mean_roc = 0
            else:
                mean_roc = np.mean(mean_roc)


            # Plot curve
            """ TODO FIX
            plt.step(mean_recall, mean_precision, color='b', alpha=0.2,
                     where='post')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Precision-Recall curve')
            plt.savefig('PR.jpg')
            PR_fig = np.asarray(Image.open('PR.jpg'))
            self.log_images(tag="Precision-Recall", images=[PR_fig], step=epoch)
            """

            #Add AUC
            summary = tf.Summary()
            summary.value.add(tag='AUC', simple_value=mean_roc)
            self.val_writer.add_summary(summary, epoch)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
