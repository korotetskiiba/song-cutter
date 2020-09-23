from Pipeline.Segmentation.SegmentationModule import SegmentationModule as sg
import os
import pickle

PATH_TO_DATASET = os.path.join('auxiliary_files', 'dataset', 'live_set_genres.pkl')
PATH_TO_CHECKPOINT = os.path.join('auxiliary_files', 'checkpoints', 'baseline_model_30.h5')
PATH_TO_EVAL = os.path.join('auxiliary_files', 'eval')


if __name__ == "__main__":
    # load LiveSet
    with open(PATH_TO_DATASET, "rb") as handle:
        dataset_dict = pickle.load(handle)

    x, y, _ = dataset_dict['train']  # the last tuple unit is genre labels
    x_val, y_val, _ = dataset_dict['valid']
    x_test, y_test, _ = dataset_dict['test']

    model = sg()

    model.load_from_checkpoint(PATH_TO_CHECKPOINT)
    model.evaluate(x_val, y_val, os.path.join(PATH_TO_EVAL, 'baseline'))
