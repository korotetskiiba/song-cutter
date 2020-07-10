import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from astropy.convolution import Gaussian1DKernel, convolve
import datetime

import SegmSubmodules.Models as Models
import SegmSubmodules.Evalutation as Eval
import SegmSubmodules.PredictionCutter as Cutter


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
        y_tr_crf, y_val_crf = Models.data_to_crf(y_train, y_valid)  # convert to CRF-suitable format

        callback_list = self.__define_callback_list(checkpoint_file)

        self.model.fit(x_train, y_tr_crf, batch_size=batch_size, validation_data=(x_valid, y_val_crf), epochs=epochs,
                       callbacks=callback_list)

    def load_from_checkpoint(self, checkpoint_file):
        """Loads trained model into self.model from checkpoint file

        Args:
            checkpoint_file: file *.h5 with model weights"""
        assert isinstance(checkpoint_file, str) and checkpoint_file.endswith('.h5'), "Wrong checkpoint file argument"
        self.model = Models.load_from_cpt(checkpoint_file)

    def predict(self, x_data):
        """Makes prediction on x_data. Model should be loaded or trained before predict called

        Args:
             x_data: tensor of embeddings to make prediction

        Returns:
            the list of time intervals where time interval is a list of the beginning time and the end time,
            time stored in strings in format: 'hh:mm:ss'
        """
        assert self.model is not None, "Can't predict, model has not been loaded yet"
        assert len(x_data.shape) == 3, "X_data shape must be (samples, duration, embeddings)"
        # self.model must not be None
        sample_mask = self.__get_raw_prediction(x_data)  # firstly, get raw prediction
        smooth_mask = self.__get_smooth_mask(sample_mask)  # smooth mask
        absolute_intervals = self.__get_intervals_by_mask(smooth_mask)  # convert mask to embedding intervals
        time_intervals = self.__abs_intervals_to_time(absolute_intervals)  # convert intervals to time intervals
        return time_intervals

    def get_model(self):
        """Get current trained or loaded model. For example, this may be used to show model summary

        Returns:
            keras model stored in self.model"""
        return self.model

    @staticmethod
    def cut_wav(path_to_wav, target_path, prediction_intervals):
        """Cut wav file according to time intervals given by prediction. Pieces of sound file will be saved as
        target_path + '_piece_' + piece_number + '.wav'

        Args:
             path_to_wav: path to .wav sound file to be cut
             target_path: path to directory where pieces of sound file will be saved
             prediction_intervals: the list of intervals where interval is the list of strings (the beginning
             time and the end time in format 'hh:mm:ss')
        """
        assert isinstance(path_to_wav, str) and path_to_wav.endswith(".wav"), "Wrong path to wav-file"
        Cutter.cut_file(path_to_wav, target_path, prediction_intervals)

    @staticmethod
    def cut_video(path_to_video, target_path, prediction_intervals):
        """Cut video file according to time intervals given by prediction. Pieces of video file will be saved as
                target_path + '_piece_' + piece_number + '.mp4'

                Args:
                     path_to_video: path to .mp4 video file to be cut
                     target_path: path to directory where pieces of video file will be saved
                     prediction_intervals: the list of intervals where interval is the list of strings (the beginning
                     time and the end time in format 'hh:mm:ss')
                """
        assert isinstance(path_to_video, str) and path_to_video.endswith(".mp4"), "Wrong path to wav-file"
        Cutter.cut_file(path_to_video, target_path, prediction_intervals, ".mp4")

    def evaluate(self, x_test, y_test, target_path):
        """Evaluate trained or loaded model on test data. Saves plots and metrics to the target_path

        Args:
             x_test: test data tensor of embeddings
             y_test: ground truth
             target_path: directory where plots and metrics will be saved
        """
        assert len(x_test.shape) == 3, "X_test shape must be (samples, duration, embeddings)"
        assert len(y_test.shape) == 1, "Y_test shape must be (duration, )"
        roc_fname = target_path + "\\roc_curve.png"  # create names for artifacts
        mask_plot_fname = target_path + "\\masks.png"
        metrics_fname = target_path + "\\metrics.json"

        sample_mask = self.__get_raw_prediction(x_test)  # raw prediction
        smooth_mask = self.__get_smooth_mask(sample_mask)  # smooth prediction

        # create all reports
        Eval.count_metrics_on_sample(smooth_mask, y_test, metrics_fname)  # count metrics
        Eval.draw_roc(sample_mask, smooth_mask, y_test, roc_fname)  # draw ROC curve
        Eval.draw_mask_plots(smooth_mask, y_test, mask_plot_fname)  # draw plot of prediction and ground truth

    # private:
    def __build_new_model(self):
        """Build default model with CRF as the last layer and stores it in self.model"""
        self.model = Models.build_model()

    def __get_raw_prediction(self, x_data):
        """Gets raw prediction (binary mask vector corresponding to embeddings)

        Args:
            x_data: data tensor of embeddings to make predictions

        Returns:
            binary mask vector of predictions"""
        mask = Models.predict_mask_long_sample(x_data, self.model)
        # invert mask
        mask = [1 - v for v in mask]
        return mask

    @staticmethod
    def __get_smooth_mask(sample_mask):
        """Applies filter to the raw mask to smooth it

        Args:
            sample_mask: binary mask vector of predictions

        Returns:
            smooth mask (binary mask vector)"""
        # Create kernel
        g_kernel = Gaussian1DKernel(stddev=5)

        # Convolve data
        smooth_mask = convolve(sample_mask, g_kernel, boundary='extend')

        # to binary mask
        round_border = 0.3  # if value is higher, it's musical embedding
        smooth_mask = [int(t > round_border) for t in smooth_mask]
        return smooth_mask

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
        fragments_list = []
        start_segment = idxs[0]
        finish_segment = idxs[0]
        prev_val = idxs[0]
        for v in idxs:  # unite neighbor mask ones into intervals
            if v - prev_val < max_lag:
                prev_val = v
                finish_segment = v
            else:
                if finish_segment - start_segment > 10:
                    fragments_list.append([start_segment, finish_segment])
                    start_segment = v
                    finish_segment = v
                    prev_val = v
                else:
                    start_segment = v
                    finish_segment = v
                    prev_val = v
        if finish_segment - start_segment > 10:
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
    def __define_callback_list(checkpoint_file):
        """Get callbacks list to pass it to the fit. Callbacks include learning rate scheduler, checkpoint and
        early stopping

        Args:
            checkpoint_file: path to file where model checkpoint will be saved

        Returns:
            the list of callbacks which can be passed to fit as callback parameter instantly"""
        # define learning rate schedule
        lr_scheduler_callback = LearningRateScheduler(lambda epoch: 0.001 if epoch < 3 else 0.0003)

        # define checkpoints
        cpt_callback = ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=0, save_best_only=True,
                                       save_weights_only=False, mode='auto')

        # define early stopping
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        return [cpt_callback, lr_scheduler_callback, early_stopping_callback]
