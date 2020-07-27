from Pipeline.DataGenerator.DataGenerator import DataGenerator as dg
from Pipeline.DataGenerator.DataGenerator import KindOfData as kd
from Pipeline.Segmentation.SegmentationModule import SegmentationModule as sg
import os
import pickle


def full_fit_pipeline():
    path_to_liveset = os.path.join('auxiliary_files', 'dataset', 'live_set_genres.pkl')

    segmentator = init_model()
    fit_model_audioset(segmentator)
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

    history = segmentator.exec_fit(x_train, x_valid, y_train, y_valid, checkpoints_file, epochs=0)  # early stopping

    # save history
    history_file = os.path.join('auxiliary_files', 'history', 'fullfit')
    if not os.path.isdir(history_file):
        os.makedirs(history_file, exist_ok=True)
    history_file = os.path.join(history_file, 'pretrain.txt')
    with open(history_file, 'w') as f:
        print(history.history, file=f)

    segmentator.evaluate(x_valid, y_valid, metrics_dir)


def fix_model_params(segmentator, need_show_summary=True):
    model = segmentator.get_model()
    DENSE_LAYER_POS = 1
    model.layers[DENSE_LAYER_POS].trainable = False  # fix params
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

    history = segmentator.exec_fit(x, x_val, y, y_val, checkpoints_file, epochs=0)  # early stopping

    # save history
    history_file = os.path.join('auxiliary_files', 'history', 'fullfit')
    if not os.path.isdir(history_file):
        os.makedirs(history_file, exist_ok=True)
    history_file = os.path.join(history_file, 'posttrain.txt')
    with open(history_file, 'w') as f:
        print(history.history, file=f)

    segmentator.evaluate(x_val, y_val, metrics_dir)


if __name__ == "__main__":
    full_fit_pipeline()
