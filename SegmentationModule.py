import numpy as np
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from astropy.convolution import Gaussian1DKernel, convolve
import datetime

import SegmSubmodules.Models as Models
import SegmSubmodules.Evalutation as Eval
import SegmSubmodules.PredictionCutter


class SegmentationModule:

    # public:
    def __init__(self):
        self.model = None

    def exec_fit(self, x_train, x_valid, y_train, y_valid, checkpoint_file, epochs=30, batch_size=32):
        if self.model is None:
            self.__build_new_model()
        y_tr_crf, y_val_crf = self.__data_to_crf(y_train, y_valid)

        callback_list = self.__define_callback_list(checkpoint_file)

        self.model.fit(x_train, y_tr_crf, batch_size=batch_size, validation_data=(x_valid, y_val_crf), epochs=epochs,
                       callbacks=callback_list)

    def predict(self, x_data, checkpoint_file=None):
        if self.model is None:
            self.model = Models.load_from_cpt(checkpoint_file)
        sample_mask = self.__get_raw_prediction(x_data)
        smooth_mask = self.__get_smooth_mask(sample_mask)
        absolute_intervals = self.__get_intervals_by_mask(smooth_mask)
        time_intervals = self.__abs_intervals_to_time(absolute_intervals)
        return time_intervals


    @staticmethod
    def cut_wav(path_to_wav, target_path, prediction_intervals):
        # Not implemented yet
        pass

    @staticmethod
    def cut_video(path_to_video, target_path, prediction_intervals):
        # Not implemented yet
        pass

    def evaluate(self, x_test, y_test, target_path):
        roc_fname = target_path + "\\roc_curve.png"
        mask_plot_fname = target_path + "\\masks.png"
        metrics_fname = target_path + "\\metrics.json"

        sample_mask = self.__get_raw_prediction(x_test)
        smooth_mask = self.__get_smooth_mask(sample_mask)

        # create all reports
        Eval.count_metrics_on_sample(smooth_mask, y_test, metrics_fname)
        Eval.draw_roc(sample_mask, smooth_mask, y_test, roc_fname)
        Eval.draw_mask_plots(smooth_mask, y_test, mask_plot_fname)

    # private:
    def __build_new_model(self):
        self.model = Models.build_model()

    @staticmethod
    # this method may move to the Models.SegmCRFModelCreator
    def __data_to_crf(y_tr, y_val):
        # convert data for CRF format
        crf_y_tr = [to_categorical(i, num_classes=2) for i in y_tr]
        crf_y_val = [to_categorical(i, num_classes=2) for i in y_val]

        crf_y_tr = np.array(crf_y_tr)
        crf_y_val = np.array(crf_y_val)
        return crf_y_tr, crf_y_val

    def __get_raw_prediction(self, x_data):
        return Models.predict_mask_long_sample(x_data, self.model)

    @staticmethod
    def __get_smooth_mask(sample_mask):
        # Create kernel
        g_kernel = Gaussian1DKernel(stddev=5)

        # Convolve data
        smooth_mask = convolve(sample_mask, g_kernel, boundary='extend')

        # invert mask
        smooth_mask = [1 - v for v in smooth_mask]

        # to binary mask
        round_border = 0.3
        smooth_mask = [int(t > round_border) for t in smooth_mask]
        return smooth_mask

    @staticmethod
    def __get_intervals_by_mask(mask):
        idxs = np.nonzero(mask)[0]

        max_lag = 4
        fragments_list = []
        start_segment = idxs[0]
        finish_segment = idxs[0]
        prev_val = idxs[0]
        for v in idxs:
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
        sec_to_time = lambda arg: str(datetime.timedelta(seconds=arg))
        vgg_rescale = 0.96  # 100 embeddings of VGG is 96 seconds
        intervals = [[sec_to_time(int(vgg_rescale * q)) for q in v] for v in abs_intervals]
        return intervals

    @staticmethod
    def __define_callback_list(checkpoint_file):
        # define learning rate schedule
        lr_scheduler_callback = LearningRateScheduler(lambda epoch: 0.001 if epoch < 3 else 0.0003)

        # define checkpoints
        cpt_callback = ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=0, save_best_only=True,
                                       save_weights_only=False, mode='auto')

        # define early stopping
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        return [cpt_callback, lr_scheduler_callback, early_stopping_callback]
