import os

from keras.callbacks import TensorBoard, EarlyStopping

from tensorboard.plugins.pr_curve import summary as pr_summary
from tensorboard.plugins.scalar import summary as sc_summary
from tensorboard.plugins.image import summary as im_summary
from image_processing import create_patches_from_images, preprocess_image
from utils import normalize_numpy
from predict import predict
import numpy as np
import tensorflow as tf
import skimage.io
from io import StringIO,BytesIO
from sklearn.metrics import roc_auc_score
from create_datasets import load_data_for_patient
from tqdm import tqdm
import keras.backend as K

class PatientPlotCallback(TensorBoard):
    def __init__(self, patients=None, patch_size=None, folders_input=["T2"], folders_target=["lesion"], verbose=0, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        self.val_log_dir = os.path.join(log_dir, 'validation')

        super(PatientPlotCallback, self).__init__(training_log_dir, **kwargs)

        # Get PR Curve
        self.patch_size = patch_size
        self.verbose = verbose
        self.patients = patients
        self.folders_input = folders_input
        self.folders_target = folders_target

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(PatientPlotCallback, self).set_model(model)

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

        # add image
        self.__log_example_image(epoch)
        self.val_writer.flush()
        self.writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(PatientPlotCallback, self).on_epoch_end(epoch, logs)

    def log_images(self, tag, images, step, writer):
        """Logs a list of images."""
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

    def __log_example_image(self, epoch):

        patients = self.patients
        for patient, [type_writer, layers] in patients.items():
            # select writer; default is train
            writer = self.writer
            if type_writer=="validation":
                writer = self.val_writer

            MTT, CBF, CBV, Tmax, T2, lesion = load_data_for_patient(patient)

            dict_inputs = {"MTT": MTT,
                           "CBF": CBF,
                           "CBV": CBV,
                           "Tmax": Tmax,
                           "T2": T2,
                           "lesion": lesion}

            print("Shapes of data : ")
            for k in dict_inputs.keys():
                print("\t", k, " - ", dict_inputs[k].shape)

            images_input = [dict_inputs[k] for k in self.folders_input]
            images_target = [dict_inputs[k] for k in self.folders_target]

            # Predict the output.
            predicted_image = predict(images_input, self.model, self.patch_size, verbose=self.verbose)

            for layer in layers:

                pred_image = predicted_image[:, :, layer]
                image_original = images_input[0][:, :, layer]
                lesion_original = images_target[0][:, :, layer]

                # RGB
                merged_image = np.zeros([pred_image.shape[0], pred_image.shape[1], 3])
                merged_image[:, :, 0] = lesion_original
                merged_image[:, :, 1] = pred_image
                merged_image[:, :, 2] = image_original

                self.log_images(tag="Prediction example (layer{}), with first channel of inputs and lesions".format(layer),
                                images=[pred_image, lesion_original, image_original, merged_image], step=epoch,
                                writer=writer)


    def on_train_end(self, logs=None):
        super(PatientPlotCallback, self).on_train_end(logs)
        self.val_writer.close()
