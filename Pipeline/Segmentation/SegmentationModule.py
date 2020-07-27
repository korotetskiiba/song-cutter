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

    def exec_fit(self, x_train, x_valid, y_train, y_valid, checkpoint_file, epochs=30, batch_size=32,
                 callback_list=None):
        """Creates model and launches fit. Trained model stored in self.model

        Args:
            x_train: train data tensor (number of samples, duration, embedding)
            x_valid: validation data tensor
            y_train: train data labels tensor
            y_valid: validation data labels tensor
            checkpoint_file: name of file where checkpoints will be saved
            epochs: the number of epochs to fit, set 0 to train until early stopping callback used
            batch_size: the number of samples in a single batch to fit
        Returns:
            model history (the same return as keras model fit)"""
        assert len(x_train.shape) == 3, "X_train shape must be (samples, time, embeddings)"
        assert len(x_valid.shape) == 3, "X_valid shape must be (samples, time, embeddings)"
        assert len(y_train.shape) == 3, "Y_train shape must be (samples, time, 1)"
        assert len(y_valid.shape) == 3, "Y_valid shape must be (samples, time, 1)"
        assert isinstance(checkpoint_file, str) and checkpoint_file.endswith(".h5"), ("Checkpoint file must be string"
                                                                                      "ends with .h5")
        if self.model is None:  # build default model
            self.__build_new_model()
        y_tr_crf, y_val_crf = Models.convert_to_crf_format(y_train, y_valid)

        callback_list = self.__define_callback_list(checkpoint_file, callback_list)

        SO_MANY_EPOCHS = 500
        if epochs == 0:  # use early stopping callback as finishing
            epochs = SO_MANY_EPOCHS

        return self.model.fit(x_train, y_tr_crf, batch_size=batch_size, validation_data=(x_valid, y_val_crf), epochs=epochs,
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
        assert len(x_data.shape) == 3, "X_data shape must be (samples=1, duration, embeddings)"
        sample_mask = self.__get_raw_prediction(x_data)  # firstly, get raw prediction
        if need_smoothing:
            sample_mask = self.__get_smooth_mask(sample_mask)  # smooth mask
        absolute_intervals = self.__get_intervals_by_mask(sample_mask[0, :])  # convert mask to embedding intervals
        time_intervals = self.__abs_intervals_to_time(absolute_intervals)  # convert intervals to time intervals
        return time_intervals

    def predict_with_genre(self, x_data, genre_duration=31, need_smoothing=True):
        """Makes prediction on x_data and returns x_data slices suitable to predict genre

                Args:
                     x_data: tensor of embeddings to make prediction
                     genre_duration: the duration of tensor to predict genre in MIE (31 embeddings == 30 seconds)
                     need_smoothing: True if it is needed to smooth mask after getting raw prediction mask

                Returns:
                    tuple (time_intervals, x_genres_embed) where
                    time_intervals: the list of time intervals where time interval is a list of the beginning time and
                                    the end time, time stored in strings in format: 'hh:mm:ss'
                    x_genres_embed: tensor of shape (segmentated_parts, genre_duration, embed) of embeddings ready for
                                    MIE predict
                """
        assert self.model is not None, "Can't predict, model has not been loaded yet"
        assert len(x_data.shape) == 3, "X_data shape must be (samples=1, duration, embeddings)"
        sample_mask = self.__get_raw_prediction(x_data)  # firstly, get raw prediction
        if need_smoothing:
            sample_mask = self.__get_smooth_mask(sample_mask)  # smooth mask
        absolute_intervals = self.__get_intervals_by_mask(sample_mask[0, :])  # convert mask to embedding intervals
        time_intervals = self.__abs_intervals_to_time(absolute_intervals)  # convert intervals to time intervals

        x_genres_list = []
        for current_interval in absolute_intervals:  # transform each piece to an independent sample
            if current_interval[1] - current_interval[0] + 1 >= genre_duration:  # too many embeddings for MIE
                x_genres_list.append(x_data[0, current_interval[0]:(current_interval[0] + genre_duration), :])
            else:  # too less embeddings for MIE
                current_part = x_data[0, current_interval[0]:current_interval[1], :]
                x_genres_sample = np.zeros(shape=(genre_duration, x_data.shape[2]), dtype=np.float32)
                cur_part_len = current_part.shape[0]
                parts_count = int(genre_duration // cur_part_len)
                for i in range(parts_count):  # copy pieces parts_count times
                    x_genres_sample[i * cur_part_len: (i + 1) * cur_part_len, :] = current_part
                x_genres_sample[parts_count * cur_part_len:] = \
                    current_part[:(genre_duration - parts_count * cur_part_len), :]  # last piece
                x_genres_list.append(x_genres_sample)

        x_genres_embed = np.array(x_genres_list)
        return time_intervals, x_genres_embed

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

    def evaluate(self, x_test, y_test, target_path, genres_mask=None, genres_dict=None, genres_clamp=31,
                 plot_time_clamp=1000):
        """Evaluate trained or loaded model on test data. Saves plots and metrics to the target_path

        Args:
             x_test: test data tensor of embeddings
             y_test: test data tensor of masks
             target_path: directory where plots and metrics will be saved
             genres_mask: data tensor of genres masks (may be None)
             genres_dict: dict with keys as numbers of genre classes and values as corresponding MIE genres (strings)
             genres_clamp: length of piece suitable for MIE to predict genre (31 embeddings == 30 seconds)
             plot_time_clamp: the duration of the part to make plot mask (from the beginning)

        Returns:
            tuple (x_genres, y_genres) where
            x_genres: segmentated samples to predict genres (suitable for MIE)
            y_genres: corresponding masks to predict genres (suitable for MIE)
        """
        assert len(x_test.shape) == 3, "X_test shape must be (samples, duration, embeddings)"
        assert len(y_test.shape) == 3, "Y_test shape must be (samples, duration, 1)"
        assert genres_mask is None or len(genres_mask.shape) == 3, "Genres mask shape must be " \
                                                                   "(samples, duration, genre)"
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

        if genres_mask is not None:
            x_genres, y_genres = self.__transform_genres_mask(x_test, smooth_mask, genres_mask,
                                                              genres_dict, genres_clamp)
            return x_genres, y_genres

    # private:
    def __transform_genres_mask(self, x, pred, g_mask, genres_dict, genres_clamp=31):
        """Transforms X and prediction into embeddings of genres_clamp length to pass to MIE module

        Args:
            x: input tensor of shape (samples, time, embed)
            pred: prediction for the x tensor (binary mask tensor of shape (samples, time))
            g_mask: ground truth, the genres mask tensor of shape (samples, time, genre)
            genres_dict: dictionary with genre numbers as keys and values suitable for MIE
            genres_clamp: the duration of part suitable for MIE (31 embeddings is 30 seconds)

        Returns:
            tuple (x_genres_array, y_genre_labels) where
            x_genres_array: embedding tensor of shape (samples, genres_clamp, embed)
            y_genre_labels: list of genres corresponding to each x_genres_array sample, suitable for MIE
        """
        x_genres_list = []
        y_genre_labels = []
        for cur_sample in range(x.shape[0]):  # for all samples
            absolute_intervals = self.__get_intervals_by_mask(pred[cur_sample, :].reshape(-1))  # mask to embedding intervals
            for current_interval in absolute_intervals:  # transform each piece to an independent sample
                if current_interval[1] - current_interval[0] + 1 >= genres_clamp:  # too many embeddings for MIE
                    x_genres_list.append(x[cur_sample, current_interval[0]:(current_interval[0] + genres_clamp), :])
                    y_label_number = int(np.median(g_mask[cur_sample, current_interval[0]:(current_interval[0] + genres_clamp), :]))
                    y_genre_labels.append(genres_dict[y_label_number])
                else:  # too less embeddings for MIE
                    current_part = x[cur_sample, current_interval[0]:current_interval[1], :]
                    x_genres_sample = np.zeros(shape=(genres_clamp, x.shape[2]), dtype=np.float32)
                    cur_part_len = current_part.shape[0]
                    parts_count = int(genres_clamp // cur_part_len)
                    for i in range(parts_count):  # copy pieces parts_count times
                        x_genres_sample[i * cur_part_len: (i + 1) * cur_part_len, :] = current_part
                    x_genres_sample[parts_count * cur_part_len:] = \
                        current_part[:(genres_clamp - parts_count * cur_part_len), :]  # last piece
                    x_genres_list.append(x_genres_sample)
                    y_label_number = np.median(g_mask[cur_sample, current_interval[0]:current_interval[1], :])
                    y_genre_labels.append(genres_dict[y_label_number])

        x_genres_array = np.array(x_genres_list)  # list to numpy array
        return x_genres_array, y_genre_labels

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
        if len(idxs) == 0:
            return []

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
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=0.0008)

        if custom_callback_list is None:
            result_cb_list = [cpt_callback, lr_scheduler_callback, early_stopping_callback]
        else:
            result_cb_list = custom_callback_list

        return result_cb_list


if __name__ == "__main__":
    CommandLineHelper.run_from_cmd(SegmentationModule())
