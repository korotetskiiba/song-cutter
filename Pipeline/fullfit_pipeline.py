from Pipeline.DataGenerator.DataGenerator import DataGenerator as dg
from Pipeline.DataGenerator.DataGenerator import KindOfData as kd
from Pipeline.Segmentation.SegmentationModule import SegmentationModule as sg
import os
import pickle
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import numpy as np


def full_fit_pipeline(experiment_name):  # fit on AudioSet, fix GRU & Dense, fit on LiveSet
    path_to_liveset = os.path.join('auxiliary_files', 'dataset', 'live_set_genres.pkl')

    segmentator = sg()  # init model
    fit_model_audioset(segmentator, experiment_name)
    fix_model_params(segmentator)
    fit_model_live(segmentator, path_to_liveset, experiment_name)


def dual_fit(experiment_name):  # fit on AudioSet, don't fix any weights, fit on LiveSet
    path_to_liveset = os.path.join('auxiliary_files', 'dataset', 'live_set_genres.pkl')

    segmentator = sg()  # init model
    fit_model_audioset(segmentator, experiment_name)
    fit_model_live(segmentator, path_to_liveset, experiment_name)


def fit_model_mixed_data(experiment_name):  # vstack AudioSet and LiveSet train tensors and fit
    path_to_liveset = os.path.join('auxiliary_files', 'dataset', 'live_set_genres.pkl')

    # load audioset
    sets = dg.get_generated_sample(kd.AUDIOSET, [7, 2, 1])
    x_tr_a, y_tr_a = sets['train']
    x_val_a, y_val_a = sets['val']

    # load liveset
    with open(path_to_liveset, "rb") as handle:
        dataset_dict = pickle.load(handle)
    x_tr_l, y_tr_l, _ = dataset_dict['train']  # the last tuple unit is genre labels
    x_val_l, y_val_l, _ = dataset_dict['valid']

    # concat datasets
    x_train = np.vstack([x_tr_a, x_tr_l])
    x_valid = np.vstack([x_val_a, x_val_l])
    y_train = np.vstack([y_tr_a, y_tr_l])
    y_valid = np.vstack([y_val_a, y_val_l])

    # checkpoints
    checkpoints_file = os.path.join('auxiliary_files', 'checkpoints', 'fullfit', experiment_name)
    if not os.path.isdir(checkpoints_file):
        os.makedirs(checkpoints_file, exist_ok=True)

    checkpoints_file = os.path.join(checkpoints_file, 'cpt.h5')

    callback_list = mix_model_callbacks(checkpoints_file)  # get callbacks for this model
    segmentator = sg()  # init model

    history = segmentator.exec_fit(x_train, x_valid, y_train, y_valid, checkpoints_file,
                                   epochs=0, callback_list=callback_list)  # 0 epochs for early stopping

    # save history
    history_file = os.path.join('auxiliary_files', 'history', 'new', experiment_name)
    if not os.path.isdir(history_file):
        os.makedirs(history_file, exist_ok=True)
    history_file = os.path.join(history_file, 'history.txt')
    with open(history_file, 'w') as f:
        print(history.history, file=f)

    metrics_dir = os.path.join('auxiliary_files', 'eval', 'new', experiment_name, 'metrics')
    if not os.path.isdir(metrics_dir):
        os.makedirs(metrics_dir, exist_ok=True)

    segmentator.evaluate(x_val_l, y_val_l, metrics_dir, plot_time_clamp=2000)


def fit_model_mix_shuffle(experiment_name):  # mix folds of AudioSet and LiveSet train tensors and fit
    path_to_liveset = os.path.join('auxiliary_files', 'dataset', 'live_set_genres.pkl')

    # load audioset
    sets = dg.get_generated_sample(kd.AUDIOSET, [7, 2, 1])
    x_tr_a, y_tr_a = sets['train']
    x_val_a, y_val_a = sets['val']

    # load liveset
    with open(path_to_liveset, "rb") as handle:
        dataset_dict = pickle.load(handle)

    x_tr_l, y_tr_l, _ = dataset_dict['train']  # the last tuple unit is genre labels
    x_val_l, y_val_l, _ = dataset_dict['valid']

    # concat datasets
    folds_count = 10
    x_train, y_train = mix_folds(x_tr_a, x_tr_l, y_tr_a, y_tr_l, folds_count)
    x_valid, y_valid = mix_folds(x_val_a, x_val_l, y_val_a, y_val_l, folds_count)

    # checkpoints
    checkpoints_file = os.path.join('auxiliary_files', 'checkpoints', 'fullfit', experiment_name)
    if not os.path.isdir(checkpoints_file):
        os.makedirs(checkpoints_file, exist_ok=True)

    checkpoints_file = os.path.join(checkpoints_file, 'cpt.h5')

    callback_list = mix_model_callbacks(checkpoints_file)
    segmentator = sg()  # init model

    history = segmentator.exec_fit(x_train, x_valid, y_train, y_valid, checkpoints_file,
                                   epochs=0, callback_list=callback_list)  # 0 epochs for early stopping

    # save history
    history_file = os.path.join('auxiliary_files', 'history', 'new', experiment_name)
    if not os.path.isdir(history_file):
        os.makedirs(history_file, exist_ok=True)
    history_file = os.path.join(history_file, 'history.txt')
    with open(history_file, 'w') as f:
        print(history.history, file=f)

    metrics_dir = os.path.join('auxiliary_files', 'eval', 'new', experiment_name, 'metrics')
    if not os.path.isdir(metrics_dir):
        os.makedirs(metrics_dir, exist_ok=True)

    segmentator.evaluate(x_val_l, y_val_l, metrics_dir, plot_time_clamp=2000)


# used to split data tensors into folds and merge them one by one into a single tensor
def mix_folds(x_tr_a, x_tr_l, y_tr_a, y_tr_l, folds_cnt):
    train_live_len = x_tr_l.shape[0]  # length of the LiveSet part
    train_audi_len = x_tr_a.shape[0]  # length of the AudioSet part

    # allocate tensors
    x_train = np.zeros(shape=(train_live_len + train_audi_len, x_tr_l.shape[1], x_tr_l.shape[2]),
                       dtype=np.float32)
    y_train = np.zeros(shape=(train_live_len + train_audi_len, y_tr_l.shape[1], y_tr_l.shape[2]),
                       dtype=np.float32)

    audio_fold_duration = train_audi_len // folds_cnt  # duration of a single fold of AudioSet
    live_fold_duration = train_live_len // folds_cnt  # duration of a single fold of LiveSet
    mix_fold_duration = audio_fold_duration + live_fold_duration  # duration of resulting merged fold
    for k_fold in range(folds_cnt):
        # place train fold
        x_train[k_fold * mix_fold_duration: k_fold * mix_fold_duration + audio_fold_duration, :, :] = \
            x_tr_a[k_fold * audio_fold_duration: (k_fold + 1) * audio_fold_duration, :, :]  # place AudioSet fold
        x_train[k_fold * mix_fold_duration + audio_fold_duration: (k_fold + 1) * mix_fold_duration, :, :] = \
            x_tr_l[k_fold * live_fold_duration: (k_fold + 1) * live_fold_duration, :, :]  # place LiveSet fold

        # place validation fold
        y_train[k_fold * mix_fold_duration: k_fold * mix_fold_duration + audio_fold_duration, :, :] = \
            y_tr_a[k_fold * audio_fold_duration: (k_fold + 1) * audio_fold_duration, :, :]  # place AudioSet fold
        y_train[k_fold * mix_fold_duration + audio_fold_duration: (k_fold + 1) * mix_fold_duration, :, :] = \
            y_tr_l[k_fold * live_fold_duration: (k_fold + 1) * live_fold_duration, :, :]  # place LiveSet fold

    # merge tails
    remaining_audio = train_audi_len - folds_cnt * audio_fold_duration
    remaining_live = train_live_len - folds_cnt * live_fold_duration
    # place AudioSet tail
    if remaining_audio != 0:
        x_train[mix_fold_duration * folds_cnt: mix_fold_duration * folds_cnt + remaining_audio, :, :] = \
            x_tr_a[-remaining_audio:, :, :]
        y_train[mix_fold_duration * folds_cnt: mix_fold_duration * folds_cnt + remaining_audio, :, :] = \
            y_tr_a[-remaining_audio:, :, :]

    # place LiveSet tail
    if remaining_live != 0:
        x_train[mix_fold_duration * folds_cnt + remaining_audio:, :, :] = \
            x_tr_l[-remaining_live:, :, :]
        y_train[mix_fold_duration * folds_cnt + remaining_audio:, :, :] = \
            y_tr_l[-remaining_live:, :, :]

    return x_train, y_train


# fit model held in segmentator on AudioSet
def fit_model_audioset(segmentator, experiment_name):
    # load audioset
    sets = dg.get_generated_sample(kd.AUDIOSET, [7, 2, 1])
    x_train, y_train = sets['train']
    x_valid, y_valid = sets['val']

    # create dirs
    checkpoints_file = os.path.join('auxiliary_files', 'checkpoints', 'fullfit', experiment_name)
    if not os.path.isdir(checkpoints_file):
        os.makedirs(checkpoints_file, exist_ok=True)

    checkpoints_file = os.path.join(checkpoints_file, 'pretrain.h5')

    metrics_dir = os.path.join('auxiliary_files', 'eval', 'fullfit', experiment_name, 'pretrain')
    if not os.path.isdir(metrics_dir):
        os.makedirs(metrics_dir, exist_ok=True)

    callback_list = audioset_callbacks(checkpoints_file)
    history = segmentator.exec_fit(x_train, x_valid, y_train, y_valid, checkpoints_file,
                                   epochs=0, callback_list=callback_list)  # 0 epochs for early stopping

    # save history
    history_file = os.path.join('auxiliary_files', 'history', 'fullfit', experiment_name)
    if not os.path.isdir(history_file):
        os.makedirs(history_file, exist_ok=True)
    history_file = os.path.join(history_file, 'pretrain.txt')
    with open(history_file, 'w') as f:
        print(history.history, file=f)

    segmentator.evaluate(x_valid, y_valid, metrics_dir)


# fix parameters of the model held in segmentator
def fix_model_params(segmentator, need_show_summary=True):
    model = segmentator.get_model()
    layers_to_fix = [1, 2, 3]  # 1 is GRU, 2 is Dropout, 3 is Dense
    for layer in layers_to_fix:
        model.layers[layer].trainable = False  # fix params
    if need_show_summary:
        print("Model layers:")
        for layer in model.layers:
            print(layer.name, layer.trainable)
        model.summary()


# fit model held in segmentator on LiveSet
def fit_model_live(segmentator, path_to_liveset, experiment_name):
    # load datasets
    with open(path_to_liveset, "rb") as handle:
        dataset_dict = pickle.load(handle)

    x, y, _ = dataset_dict['train']  # the last tuple unit is genre labels
    x_val, y_val, _ = dataset_dict['valid']

    checkpoints_file = os.path.join('auxiliary_files', 'checkpoints', 'fullfit', experiment_name)
    if not os.path.isdir(checkpoints_file):
        os.makedirs(checkpoints_file, exist_ok=True)

    checkpoints_file = os.path.join(checkpoints_file, 'posttrain.h5')

    metrics_dir = os.path.join('auxiliary_files', 'eval', 'fullfit', experiment_name, 'posttrain')
    if not os.path.isdir(metrics_dir):
        os.makedirs(metrics_dir, exist_ok=True)

    callback_list = liveset_callbacks(checkpoints_file)
    history = segmentator.exec_fit(x, x_val, y, y_val, checkpoints_file,
                                   epochs=0, callback_list=callback_list)  # early stopping

    # save history
    history_file = os.path.join('auxiliary_files', 'history', 'fullfit', experiment_name)
    if not os.path.isdir(history_file):
        os.makedirs(history_file, exist_ok=True)
    history_file = os.path.join(history_file, 'posttrain.txt')
    with open(history_file, 'w') as f:
        print(history.history, file=f)

    segmentator.evaluate(x_val, y_val, metrics_dir)


# CALLBACKS


def mix_model_callbacks(checkpoint_file):
    # define learning rate schedule
    lr_scheduler_callback = LearningRateScheduler(lambda epoch: 0.001 if epoch < 3 else 0.0003)

    # define checkpoints
    cpt_callback = ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=0, save_best_only=True,
                                   save_weights_only=False, mode='auto')

    # define early stopping
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=0.0008)
    return [lr_scheduler_callback, cpt_callback, early_stopping_callback]


def audioset_callbacks(checkpoint_file):
    # define learning rate schedule
    lr_scheduler_callback = LearningRateScheduler(lambda epoch: 0.001 if epoch < 3 else 0.0003)

    # define checkpoints
    cpt_callback = ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=0, save_best_only=True,
                                   save_weights_only=False, mode='auto')

    # define early stopping
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=0.0008)
    return [lr_scheduler_callback, cpt_callback, early_stopping_callback]


def liveset_callbacks(checkpoints_file):
    # define learning rate schedule
    lr_scheduler_callback = LearningRateScheduler(lambda epoch: 0.0003 if epoch < 3 else 0.0001)

    # define checkpoints
    cpt_callback = ModelCheckpoint(checkpoints_file, monitor='val_loss', verbose=0, save_best_only=True,
                                   save_weights_only=False, mode='auto')

    # define early stopping
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=0.0001)
    return [lr_scheduler_callback, cpt_callback, early_stopping_callback]


if __name__ == "__main__":
    # print("Type experiment name:")
    # experiment_name = input()
    # full_fit_pipeline(experiment_name)

    # fit models for all 4 strategies:
    fit_model_mixed_data('mix_data')  # strategy: vstack AudioSet and LiveSet data and fit
    fit_model_mix_shuffle('mix_shuffle')  # strategy: mix AudioSet and LiveSet data (mix folds) and fit
    full_fit_pipeline('fullfit')  # strategy: fit on AudioSet, fix GRU & Dense, fit on LiveSet
    dual_fit('dual_fit')  # strategy: fit on AudioSet, don't fix weights, fit on LiveSet

