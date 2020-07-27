import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from astropy.convolution import Gaussian1DKernel, convolve
import datetime
import os.path
from pathlib import Path

import Pipeline.Segmentation.SegmSubmodules.Models as Models
import Pipeline.Segmentation.SegmSubmodules.Evaluation as Eval
import Pipeline.Segmentation.SegmSubmodules.PredictionCutter as Cutter
import Pipeline.Segmentation.SegmSubmodules.CLHelper as CommandLineHelper


class SegmentationModule:
    """This class represents pipeline module responsible for segmentation"""

    # public:
    def __init__(self):
        """Initializes module"""
        self.model = None

    def exec_fit(self, x_train, x_valid, y_train, y_valid, checkpoint_file, epochs=30, batch_size=32):
        """Creates model and launches fit. Trained model stored in self.model

        Args:
            x_train: train data tensor (number of samples, duration, embedding)
            x_valid: validation data tensor
            y_train: train data labels tensor
            y_valid: validation data labels tensor
            checkpoint_file: name of file where checkpoints will be saved
            epochs: the number of epochs to fit
            batch_size: the number of samples in a single batch to fit"""
        assert len(x_train.shape) == 3, "X_train shape must be (samples, time, embeddings)"
        assert len(x_valid.shape) == 3, "X_valid shape must be (samples, time, embeddings)"
        assert len(y_train.shape) == 3, "Y_train shape must be (samples, time, 1)"
        assert len(y_valid.shape) == 3, "Y_valid shape must be (samples, time, 1)"
        assert isinstance(checkpoint_file, str) and checkpoint_file.endswith(".h5"), ("Checkpoint file must be string"
                                                                                      "ends with .h5")
        if self.model is None:  # build default model
            self.__build_new_model()
        y_tr_crf, y_val_crf = Models.convert_to_crf_format(y_train, y_valid)

        callback_list = self.__define_callback_list(checkpoint_file)

        self.model.fit(x_train, y_tr_crf, batch_size=batch_size, validation_data=(x_valid, y_val_crf), epochs=epochs,
                       callbacks=callback_list)

    def load_from_checkpoint(self, checkpoint_file):
        """Loads trained model into self.model from checkpoint file

        Args:
            checkpoint_file: file *.h5 with model weights"""
        assert isinstance(checkpoint_file, str) and checkpoint_file.endswith('.h5'), "Wrong checkpoint file argument"
        self.model = Models.load_from_cpt(checkpoint_file)

    def predict(self, x_data, need_smoothing=True):
        """Makes prediction on x_data. Model should be loaded or trained before predict called

        Args:
             x_data: tensor of embeddings to make prediction
             need_smoothing: True if it is needed to smooth mask after getting raw prediction mask

        Returns:
            the list of time intervals where time interval is a list of the beginning time and the end time,
            time stored in strings in format: 'hh:mm:ss'
        """
        assert self.model is not None, "Can't predict, model has not been loaded yet"
        assert len(x_data.shape) == 3, "X_data shape must be (samples, duration, embeddings)"
        sample_mask = self.__get_raw_prediction(x_data)  # firstly, get raw prediction
        if need_smoothing:
            sample_mask = self.__get_smooth_mask(sample_mask)  # smooth mask
        absolute_intervals = self.__get_intervals_by_mask(sample_mask[0, :])  # convert mask to embedding intervals
        time_intervals = self.__abs_intervals_to_time(absolute_intervals)  # convert intervals to time intervals
        return time_intervals

    def get_model(self):
        """Get current trained or loaded model. For example, this may be used to show model summary

        Returns:
            keras model stored in self.model"""
        return self.model

    @staticmethod
    def cut_file(path_to_file, target_path, prediction_intervals):
        """Cut file according to time intervals given by prediction. Pieces of sound file will be saved as
                target_path + '_piece_' + piece_number + extension

                Args:
                     path_to_file: path to .wav or .mp4 file to be cut (sound or video)
                     target_path: path to directory where pieces of sound file will be saved
                     prediction_intervals: the list of intervals where interval is the list of strings (the beginning
                     time and the end time in format 'hh:mm:ss')
                """
        assert isinstance(path_to_file, str) and (path_to_file.endswith(".wav") or path_to_file.endswith(".mp4")), (
            "Wrong path to file")
        extension = Path(path_to_file).suffix  # get target file extension
        Cutter.slice_file(path_to_file, target_path, prediction_intervals, extension=extension)

    def evaluate(self, x_test, y_test, target_path, plot_time_clamp=1000):
        """Evaluate trained or loaded model on test data. Saves plots and metrics to the target_path

        Args:
             x_test: test data tensor of embeddings
             y_test: test data tensor of masks
             target_path: directory where plots and metrics will be saved
             plot_time_clamp: the duration of the part to make plot mask (from the beginning)
        """
        assert len(x_test.shape) == 3, "X_test shape must be (samples, duration, embeddings)"
        assert len(y_test.shape) == 3, "Y_test shape must be (samples, duration, 1)"
        roc_fname = os.path.join(target_path, "roc_curve.png")  # create names for artifacts
        mask_plot_fname = os.path.join(target_path, "masks.png")
        metrics_fname = os.path.join(target_path, "metrics.json")

        sample_mask = self.__get_raw_prediction(x_test)  # raw prediction
        smooth_mask = self.__get_smooth_mask(sample_mask)  # smooth prediction

        ground_truth = y_test.copy().reshape(y_test.shape[0],
                                             y_test.shape[1])  # reshape to the same format as prediction

        # create all reports
        Eval.count_metrics_on_sample(smooth_mask, ground_truth, metrics_fname)  # count metrics
        # stack masks and draw plots
        sample_complete_mask = sample_mask.reshape(-1, order='C')
        smooth_complete_mask = smooth_mask.reshape(-1, order='C')
        ground_truth_complete = ground_truth.reshape(-1, order='C')
        Eval.draw_roc(sample_complete_mask, smooth_complete_mask, ground_truth_complete, roc_fname)  # draw ROC curve
        Eval.draw_mask_plots(smooth_complete_mask[0:plot_time_clamp], ground_truth_complete[0:plot_time_clamp],
                             mask_plot_fname)  # draw plot of prediction and ground truth

    # private:
    def __build_new_model(self):
        """Build default model with CRF as the last layer and stores it in self.model"""
        self.model = Models.build_model()

    def __get_raw_prediction(self, x_data):
        """Gets raw predictions

        Args:
            x_data: data tensor of embeddings to make predictions

        Returns:
            binary mask vector of predictions"""
        masks = Models.predict_track_pack(x_data, self.model)
        # invert masks
        for sample_num in range(masks.shape[0]):
            masks[sample_num, :] = [1 - v for v in masks[sample_num, :]]
        return masks

    @staticmethod
    def __get_smooth_mask(sample_mask, round_border=0.3):
        """Applies filter to the raw mask to smooth it

        Args:
            sample_mask: binary mask vector of predictions
            round_border: if the predicted probability is higher, rounds to 1 (consider it is musical embedding)

        Returns:
            smooth mask (binary mask vector)"""
        # Create kernel
        g_kernel = Gaussian1DKernel(stddev=5)
        smooth_masks = np.zeros(sample_mask.shape)  # init answer tensor

        for sample_num in range(sample_mask.shape[0]):
            # Convolve data
            smooth = convolve(sample_mask[sample_num, :], g_kernel, boundary='extend')
            # to binary mask
            smooth = [int(t > round_border) for t in smooth]
            smooth_masks[sample_num, :] = smooth
        return smooth_masks

    @staticmethod
    def __get_intervals_by_mask(mask):
        """Translate mask as prediction to the intervals (time codes)

            Args:
                mask: prediction mask vector

            Returns:
                the list of time intervals where time interval is the list of the beginning time and
                the end time. Time stored as indexes of embeddings"""
        idxs = np.nonzero(mask)[0]  # get indexes of embeddings with music

        max_lag = 4
        min_music_len = 10
        fragments_list = []
        start_segment = idxs[0]
        finish_segment = idxs[0]
        prev_val = idxs[0]
        for v in idxs:  # unite neighbor mask ones into intervals
            if v - prev_val < max_lag:
                prev_val = v
                finish_segment = v
            else:
                if finish_segment - start_segment > min_music_len:
                    fragments_list.append([start_segment, finish_segment])
                    start_segment = v
                    finish_segment = v
                    prev_val = v
                else:
                    start_segment = v
                    finish_segment = v
                    prev_val = v
        if finish_segment - start_segment > min_music_len:
            fragments_list.append([start_segment, finish_segment])
        return fragments_list

    @staticmethod
    def __abs_intervals_to_time(abs_intervals):
        """Translate mask with indexes of embeddings to time intervals

        Args:
            abs_intervals: the list of time intervals where time interval is the list of the beginning time and
                the end time. Time stored as indexes of embeddings

        Returns:
            the list of time intervals where time interval is the list of the beginning time and the end time. Time
            stored as string of format 'hh:mm:ss' """
        sec_to_time = lambda arg: str(datetime.timedelta(seconds=arg))  # callback to translate seconds to string
        vgg_rescale = 0.96  # 100 embeddings of VGG is 96 seconds
        # translate embedding-time intervals to time-time intervals
        intervals = [[sec_to_time(int(vgg_rescale * q)) for q in v] for v in abs_intervals]
        return intervals

    @staticmethod
    def __define_callback_list(checkpoint_file, custom_callback_list=None):
        """Get callbacks list to pass it to the fit. Callbacks include learning rate scheduler, checkpoint and
        early stopping

        Args:
            checkpoint_file: path to file where model checkpoint will be saved
            custom_callback_list: custom callbacks to use in model fit. If None, no extra custom callbacks added

        Returns:
            the list of callbacks which can be passed to fit as callback parameter instantly"""
        # define learning rate schedule
        lr_scheduler_callback = LearningRateScheduler(lambda epoch: 0.001 if epoch < 3 else 0.0003)

        # define checkpoints
        cpt_callback = ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=0, save_best_only=True,
                                       save_weights_only=False, mode='auto')

        # define early stopping
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

        result_cb_list = [cpt_callback, lr_scheduler_callback, early_stopping_callback]
        if custom_callback_list is not None:  # add custom callbacks
            result_cb_list += custom_callback_list

        return result_cb_list


if __name__ == "__main__":
    CommandLineHelper.run_from_cmd(SegmentationModule())
