from Pipeline.DataGenerator.DataGenerator import DataGenerator as dg
from Pipeline.DataGenerator.DataGenerator import KindOfData as kd
from Pipeline.Segmentation.SegmentationModule import SegmentationModule as sg
import os
import pickle
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler


def full_fit_pipeline():
    path_to_liveset = os.path.join('auxiliary_files', 'dataset', 'live_set_genres.pkl')

    segmentator = init_model()
    fit_model_audioset(segmentator)
    #segmentator.load_from_checkpoint(os.path.join('auxiliary_files', 'checkpoints', 'fullfit', 'pretrain.h5'))
    fix_model_params(segmentator)
    fit_model_live(segmentator, path_to_liveset)


def init_model():
    return sg()


def fit_model_audioset(segmentator):
    # load liveset
    sets = dg.get_generated_sample(kd.AUDIOSET, [7, 2, 1])
    x_train, y_train = sets['train']
    x_valid, y_valid = sets['val']

    # create dirs
    checkpoints_file = os.path.join('auxiliary_files', 'checkpoints', 'fullfit')
    if not os.path.isdir(checkpoints_file):
        os.makedirs(checkpoints_file, exist_ok=True)

    checkpoints_file = os.path.join(checkpoints_file, 'pretrain.h5')

    metrics_dir = os.path.join('auxiliary_files', 'eval', 'fullfit', 'pretrain')
    if not os.path.isdir(metrics_dir):
        os.makedirs(metrics_dir, exist_ok=True)

    callback_list = audioset_callbacks(checkpoints_file)
    history = segmentator.exec_fit(x_train, x_valid, y_train, y_valid, checkpoints_file,
                                   epochs=0, callback_list=callback_list)  # early stopping

    # save history
    history_file = os.path.join('auxiliary_files', 'history', 'fullfit')
    if not os.path.isdir(history_file):
        os.makedirs(history_file, exist_ok=True)
    history_file = os.path.join(history_file, 'pretrain.txt')
    with open(history_file, 'w') as f:
        print(history.history, file=f)

    segmentator.evaluate(x_valid, y_valid, metrics_dir)


def audioset_callbacks(checkpoint_file):
    # define learning rate schedule
    lr_scheduler_callback = LearningRateScheduler(lambda epoch: 0.001 if epoch < 3 else 0.0003)

    # define checkpoints
    cpt_callback = ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=0, save_best_only=True,
                                   save_weights_only=False, mode='auto')

    # define early stopping
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=0.0008)
    return [lr_scheduler_callback, cpt_callback, early_stopping_callback]


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


def fit_model_live(segmentator, path_to_liveset):
    # load datasets
    with open(path_to_liveset, "rb") as handle:
        dataset_dict = pickle.load(handle)

    x, y, _ = dataset_dict['train']  # the last tuple unit is genre labels
    x_val, y_val, _ = dataset_dict['valid']

    checkpoints_file = os.path.join('auxiliary_files', 'checkpoints', 'fullfit')
    if not os.path.isdir(checkpoints_file):
        os.makedirs(checkpoints_file, exist_ok=True)

    checkpoints_file = os.path.join(checkpoints_file, 'posttrain.h5')

    metrics_dir = os.path.join('auxiliary_files', 'eval', 'fullfit', 'posttrain')
    if not os.path.isdir(metrics_dir):
        os.makedirs(metrics_dir, exist_ok=True)

    callback_list = liveset_callbacks(checkpoints_file)
    history = segmentator.exec_fit(x, x_val, y, y_val, checkpoints_file,
                                   epochs=0, callback_list=callback_list)  # early stopping

    # save history
    history_file = os.path.join('auxiliary_files', 'history', 'fullfit')
    if not os.path.isdir(history_file):
        os.makedirs(history_file, exist_ok=True)
    history_file = os.path.join(history_file, 'posttrain.txt')
    with open(history_file, 'w') as f:
        print(history.history, file=f)

    segmentator.evaluate(x_val, y_val, metrics_dir)


def liveset_callbacks(checkpoints_file):
    # define learning rate schedule
    lr_scheduler_callback = LearningRateScheduler(lambda epoch: 0.0003 if epoch < 3 else 0.0001)

    # define checkpoints
    cpt_callback = ModelCheckpoint(checkpoints_file, monitor='val_loss', verbose=0, save_best_only=True,
                                   save_weights_only=False, mode='auto')

    # define early stopping
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=0.0001)


if __name__ == "__main__":
    full_fit_pipeline()
