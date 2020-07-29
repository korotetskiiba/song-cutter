import os
import joblib
import numpy as np

from Pipeline.DataGenerator.DataGenerator import DataGenerator as dg
from Pipeline.MetaExtraction.GenreClassifier import GenreClassifier as gc


def remove_genres_from_dataset(numbers_of_genres, set):
    x, y = set
    indices_to_remove = []
    for num in numbers_of_genres:
        for index in range(len(y)):
            mask = list(y[index])
            if mask.index(1.0) == num:
                indices_to_remove.append(index)
    list_y = [y[index] for index in range(len(y)) if index not in indices_to_remove]
    list_x = [x[index] for index in range(len(x)) if index not in indices_to_remove]
    y_new = np.array([np.array(y) for y in list_y])
    x_new = np.array([np.array(x) for x in list_x])
    return x_new, y_new


if __name__ == "__main__":
    PATH_TO_PKL = os.path.abspath("../DataGenerator/meta_files/genre_dataset.pkl")
    PATH_TO_CHECKPOINT_FILE = os.path.abspath('./model_cpt/gru_64_avg_pool_gaus_do_batch_16.h5')
    PATH_TO_ARTEFACTS = os.path.abspath("./")
    print(PATH_TO_ARTEFACTS)
    path_to_save = os.path.abspath(os.path.join(os.path.dirname(PATH_TO_PKL), "sets.pkl"))

    data_dict = joblib.load(PATH_TO_PKL)
    category_dict = data_dict['genres_dict']
    sets = dg.get_classification_data(relation=[6, 2, 2], path_to_pkl=PATH_TO_PKL, path_to_save=path_to_save)
    x_tr, y_tr = remove_genres_from_dataset([16, 14], sets['train'])
    x_val, y_val = remove_genres_from_dataset([16, 14], sets['val'])
    x_test, y_test = remove_genres_from_dataset([16, 14], sets['test'])

    # create model
    classifier = gc()
    # train model and evaluate
    classifier.exec_fit(x_tr, x_val, y_tr, y_val, PATH_TO_CHECKPOINT_FILE, category_dict, type="RNN")

    # predict
    classifier.load_from_checkpoint(PATH_TO_CHECKPOINT_FILE, category_dict)
    pred_labels = classifier.predict(x_val)

    # evaluate
    classifier.evaluate(pred_labels, y_val, PATH_TO_ARTEFACTS)



