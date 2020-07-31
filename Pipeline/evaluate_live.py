from Pipeline.Segmentation.SegmentationModule import SegmentationModule as sg
import os
import pickle

PATH_TO_DATASET = os.path.join('auxiliary_files', 'dataset', 'live_set_genres.pkl')
PATH_TO_CHECKPOINT = os.path.join('auxiliary_files', 'checkpoints', 'baseline_model_30.h5')
PATH_TO_EVAL = os.path.join('auxiliary_files', 'eval')

NEW_CHECKPOINTS_DIR = os.path.join('auxiliary_files', 'checkpoints', 'live_postfitting')

if __name__ == "__main__":

    if not os.path.isdir(NEW_CHECKPOINTS_DIR):
        os.makedirs(NEW_CHECKPOINTS_DIR, exist_ok=True)


    # load datasets
    with open(PATH_TO_DATASET, "rb") as handle:
        dataset_dict = pickle.load(handle)

    x, y, _ = dataset_dict['train']  # the last tuple unit is genre labels
    x_val, y_val, _ = dataset_dict['valid']
    x_test, y_test, _ = dataset_dict['test']

    model = sg()
    # train model and evaluate
    #for add_epochs in range(5, 11, 1):
        #model.load_from_checkpoint(PATH_TO_CHECKPOINT)
        #cur_checkpoint = NEW_CHECKPOINTS_DIR + str(add_epochs) + '.h5'
        #model.exec_fit(x, x_val, y, y_val, cur_checkpoint, epochs=add_epochs)
        #cur_eval = os.path.join(PATH_TO_EVAL, 'epochs', str(add_epochs))
        #if not os.path.isdir(cur_eval):
        #    os.makedirs(cur_eval, exist_ok=True)
        #model.evaluate(x_val, y_val, cur_eval)

    model.load_from_checkpoint(PATH_TO_CHECKPOINT)
    model.evaluate(x_val, y_val, os.path.join(PATH_TO_EVAL, 'baseline'))
