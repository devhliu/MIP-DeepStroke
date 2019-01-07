# Extends Keras' TensorBoard callback to include the Precision-Recall summary plugin.

import os

from keras.callbacks import TensorBoard, EarlyStopping

from tensorboard.plugins.pr_curve import summary as pr_summary
from tensorboard.plugins.scalar import summary as sc_summary
from utils import normalize_numpy
from predict import predict
import numpy as np
import tensorflow as tf
import skimage.io
from io import StringIO, BytesIO
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import keras.backend as K
from image_processing import clip

from create_datasets import load_data_for_patient
class TrainValTensorBoard(TensorBoard):
    def __init__(self, training_generator=None, validation_generator=None, validation_steps=None,
                 patients=None, patch_size=None, folders_input=["T2"], folders_target=["lesion"],
                 verbose=0, log_dir='./logs', stage="wcoreg_", batch_size=32, **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

        # Get PR Curve
        # self.pr_curve = kwargs.pop('pr_curve', True)
        self.pr_curve = None
        self.initialized = False
        self.validation_generator = validation_generator
        self.validation_steps = validation_steps
        self.training_generator = training_generator
        self.verbose = verbose
        self.patch_size = patch_size
        self.verbose = verbose
        self.patients = patients
        self.folders_input = folders_input
        self.folders_target = folders_target
        self.stage = stage
        self.batch_size = batch_size

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

            self.auc, ops = tf.metrics.auc(labels, predictions, curve="PR", name="auc_metric", summation_method='careful_interpolation')
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

        ## Add PR Curve
        #self.__add_pr_curve(epoch)

        # add image
        self.__log_example_image(epoch)

        if self.training_generator:
            self.__add_batch_visualization(self.training_generator, epoch, training=True)
        if self.validation_generator:
            self.__add_batch_visualization(self.validation_generator, epoch, training=False)

        self.val_writer.flush()

        # Log learning rate
        self.__log_lr(epoch)

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def __log_lr(self, epoch, logs=None):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = K.eval(self.model.optimizer.lr)
        summary_value.tag="LR"
        self.writer.add_summary(summary, epoch)

    def log_images(self, tag, images, step, writer=None):
        """Logs a list of images."""
        if writer is None:
            writer = self.val_writer

        image_summaries = []
        for image_num, image in enumerate(images):
            # Cast and write the image to a string
            #image = np.array(image,dtype=np.float32)
            #image = img_as_float(image)
            #image = normalize_numpy(image,1.0,0.0)

            image = np.nan_to_num(image)
            try:
                # Python 3.X
                s = BytesIO()
                skimage.io.imsave(s, image)
            except Exception as e:
                print("MIN = {}, MAX = {}".format(np.min(image), np.max(image)))
                raise e
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

    def __merge_images(self, images):
        # Create merged image
        nb_images = len(images)
        square = int(np.ceil(np.sqrt(nb_images)))
        if nb_images > 0:
            im_shape = images[0].shape
            while len(images) < square**2:
                # add blank images to obtain a square
                blank_image = np.zeros(im_shape)
                blank_image[:, :, 1] = 1
                images.append(blank_image)

            merged_image = np.zeros([square*im_shape[0], square*im_shape[1], im_shape[2]])
            for i in range(0, square):
                for j in range(0, square):
                    idx = i*im_shape[0]
                    idy = j*im_shape[1]
                    merged_image[idx:idx+im_shape[0], idy:idy+im_shape[1], :] = images[i*square+j]

            return merged_image
        return None

    def __log_example_image(self, epoch):

        patients = self.patients
        for patient, [type_writer, layers] in patients.items():
            # select writer; default is train
            writer = self.writer
            if type_writer=="validation":
                writer = self.val_writer

            dict_inputs = load_data_for_patient(patient, stage=self.stage)

            images_input = [dict_inputs[k] for k in self.folders_input]
            images_target = [dict_inputs[k] for k in self.folders_target]

            # Predict the output.
            predicted_image = predict(images_input, self.model, self.patch_size, verbose=self.verbose, batch_size=self.batch_size)

            for layer in layers:

                pred_image = normalize_numpy(predicted_image[:, :, layer])
                image_original = normalize_numpy(images_input[0][:, :, layer])
                lesion_original = clip(normalize_numpy(images_target[0][:, :, layer]))

                if lesion_original.shape != pred_image.shape:
                    lesion_original = lesion_original[:pred_image.shape[0], :pred_image.shape[1]]
                # RGB
                merged_image = np.zeros([pred_image.shape[0], pred_image.shape[1], 3])
                merged_image[:, :, 0] = lesion_original
                merged_image[:, :, 1] = pred_image
                merged_image[:, :, 2] = image_original

                self.log_images(tag="Prediction example (layer{}), with first channel of inputs and lesions".format(layer),
                                images=[pred_image, lesion_original, image_original, merged_image], step=epoch,
                                writer=writer)


    def __add_batch_visualization(self, generator, epoch, training=True):
        try:
            batch = generator.next() #custom generator
        except:
            batch = next(generator) #classic generator
        if training:
            t = "training"
            writer = self.writer
        else:
            writer = self.val_writer
            t = "validation"

        for c in range(batch[0].shape[3]):
            images = []
            for x, y in zip(batch[0], batch[1][-1]):
                image = x[:, :, c]
                lesion = y[:, :, 0]
                image_layer = image[:, :]
                lesion_layer = lesion[:, :]
                merged_image = np.zeros([image_layer.shape[0], image_layer.shape[1], 3])
                merged_image[:, :, 0] = lesion_layer
                merged_image[:, :, 1] = 0
                merged_image[:, :, 2] = image_layer
                images.append(merged_image)

            images_per_log = 16
            list_merged_images = []
            for i in range(0, len(images)+images_per_log, images_per_log):
                image_merged = self.__merge_images(images[i:i+images_per_log])

                if image_merged is not None:
                    list_merged_images.append(image_merged)
            if len(list_merged_images)>0:
                self.log_images(tag="Batch : {} - channel{}".format(t, c), images=list_merged_images, step=epoch, writer=writer)

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
                x, y = generator.next()

                pred_batch = self.model.predict_on_batch(x)

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
