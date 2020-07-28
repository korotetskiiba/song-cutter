import os
import joblib
import numpy as np

from Pipeline.DataGenerator.DataGenerator import DataGenerator as dg
from Pipeline.MetaExtraction.GenreClassifier import GenreClassifier as gc

if __name__ == "__main__":
    PATH_TO_PKL = "D:\current_work\!!!KURS\EPAM\song-cutter\Pipeline\DataGenerator\meta_files\genre_dataset.pkl"
    PATH_TO_CHECKPOINT_FILE = 'D:\current_work\!!!KURS\EPAM\song-cutter\model_cpt\gru_64_avg_pool_gaus_do_batch_16.h5'
    PATH_TO_ARTEFACTS = "D:\current_work\!!!KURS\EPAM\song-cutter\Pipeline\MetaExtraction"
    path_to_save = os.path.abspath(os.path.join(os.path.dirname(PATH_TO_PKL), "sets.pkl"))

    data_dict = joblib.load(PATH_TO_PKL)
    category_dict = data_dict['genres_dict']
    sets = dg.get_classification_data(relation=[6, 2, 2], path_to_pkl=PATH_TO_PKL, path_to_save=path_to_save)
    x_tr, y_tr = sets['train']
    x_val, y_val = sets['val']
    x_test, y_test = sets['test']

    # create model
    classifier = gc()
    # train model and evaluate
    # classifier.exec_fit(x_tr, x_val, y_tr, y_val, PATH_TO_CHECKPOINT_FILE, category_dict, type="RNN")

    # predict
    classifier.load_from_checkpoint(PATH_TO_CHECKPOINT_FILE, category_dict)
    pred_labels = classifier.predict(x_test)

    # evaluate
    classifier.evaluate(pred_labels, y_test, PATH_TO_ARTEFACTS)



