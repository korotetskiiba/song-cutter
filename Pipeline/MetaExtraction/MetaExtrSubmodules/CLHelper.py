import numpy as np
import argparse
import pickle


def run_from_cmd(genre_classifier_module):
    """Parse args from command line and launch module according to the command line arguments.

    Command line parameters:
        -i: path_to_pkl file with numpy tensors packed the same as DataGenerator-output;
        -c: checkpoint_file with *.h5 extension where segmentation model will be saved or from where model will be
         loaded;
        -t: target_path where all artifacts will be saved (file with metrics);
        -f: func, scenario of module work;
            values of -f:
                fit: launches fit;
                load_predict: load model from checkpoint and make prediction;
                load_evaluate: load model from checkpoint and evaluate it;
                load_cut: load model from checkpoint and slice video.

    :param genre_classifier_module: GenreClassifierModule instance.
    """
    parser = argparse.ArgumentParser(description='Genre Classifier', add_help=False)

    parser.add_argument('-i', action="store", dest="path_to_pkl", help="*.pkl file with data tensors")
    parser.add_argument('-c', action="store", dest="checkpoint_file", help="*.h5 file with model checkpoint")
    parser.add_argument('-t', action="store", dest="target_path", help="path to the target directory")
    parser.add_argument('-type', action="store", dest="type", help="type of model to use: RNN-based or CNN-based")
    parser.add_argument('-f', action="store", dest="func", help="function to launch."
                                                                "fit-to launch fit."
                                                                "load_predict-to load checkpoint and make prediction."
                                                                "load_evaluate-to load checkpoint and evaluate model.")

    args = parser.parse_args()

    if args.type is None:
        args.type = "RNN"

    # choose scenario
    if args.func == "fit":
        __fit(
            genre_classifier_module=genre_classifier_module,
            path_to_pkl=args.path_to_pkl,
            checkpoint_file=args.checkpoint_file,
            target_path=args.target_path,
            type=args.type
        )
    elif args.func == "load_predict":
        predicted_genres = __load_predict(
            genre_classifier_module=genre_classifier_module,
            path_to_pkl=args.path_to_pkl,
            checkpoint_file=args.checkpoint_file
        )
        print("The prediction genres are:")
        print(predicted_genres)
    elif args.func == "load_evaluate":
        __load_evaluate(
            genre_classifier_module=genre_classifier_module,
            path_to_pkl=args.path_to_pkl,
            checkpoint_file=args.checkpoint_file,
            target_path=args.target_path
        )
    else:
        print("Wrong func argument")


def __fit(genre_classifier_module, type, path_to_pkl, checkpoint_file, target_path):
    """
    Executes fit scenario (fit model and evaluate it).

    :param genre_classifier_module: GenreClassifierModule instance;
    :param path_to_pkl: path to file with data tensors and category dict;
    :param checkpoint_file: path to save model checkpoints;
    :param target_path: path to save evaluation summary (plots and metrics);
    :return: void.
    """
    (category_dict, (x_train, y_train), (x_val, y_val), (x_test, y_test)) = __tensors_from_pkl(path_to_pkl)
    genre_classifier_module.exec_fit(x_train, x_val, y_train, y_val, checkpoint_file, category_dict, type=type)
    #  genre_classifier_module.evaluate(x_test, y_test, target_path)


def __load_predict(genre_classifier_module, path_to_pkl, checkpoint_file):
    """
    Executes predict scenario (load model and make prediction).

    :param genre_classifier_module: GenreClassifierModule instance;
    :param path_to_pkl: path to file with data tensors;
    :param checkpoint_file: path with model checkpoint to load model;
    :return: predicted genres.
    """
    (category_dict, (_, _), (_, _), (x_test, _)) = __tensors_from_pkl(path_to_pkl)
    genre_classifier_module.load_from_checkpoint(checkpoint_file)
    predicted_genres = genre_classifier_module.predict(x_test)
    return predicted_genres


def __load_evaluate(genre_classifier_module, path_to_pkl, checkpoint_file, target_path):
    """
    Executes evaluate scenario (load model and evaluate it).

    :param genre_classifier_module: GenreClassifierModule instance;
    :param path_to_pkl: path to file with data tensors and category dict;
    :param checkpoint_file: path with model checkpoint to load model;
    :param target_path: path to save evaluation summary (plots and metrics);
    :return: void.
    """
    # TODO: implement eval method
    (category_dict, (_, _), (_, _), (x_test, y_test)) = __tensors_from_pkl(path_to_pkl)
    genre_classifier_module.load_from_checkpoint(checkpoint_file)
    genre_classifier_module.evaluate(x_test, y_test, target_path)


def genres_to_vec(category_dict, y_genres):
    inv_category_dict = {category_dict[k]: k for k in category_dict}
    target = np.zeros((len(y_genres), len(category_dict)), dtype=np.float32)
    for i in range(len(target)):
        target[i, inv_category_dict[y_genres[i]]] = 1.0
    return target


def __tensors_from_pkl(path_to_file):
    """
    Loads tensors with train, validation, test data from file generated by DataGenerator.
    Also a dict with numbers of categories is extracted({<genre name>: <genre number>}).

    :param path_to_file: *.pkl file with data tensors and category dictionary;
    :return: category dict and 3 pairs, each containing data and ground truth labels:
             (x_train, y_train), (x_val, y_val), (x_test, y_test).
    """
    with open(path_to_file, "rb") as f:
        data_dict = pickle.load(f)

    category_dict = data_dict["category_dict"]
    (x_train, y_train) = data_dict["train"]
    (x_val, y_val) = data_dict["val"]
    (x_test, y_test) = data_dict["test"]

    # labels to vecs
    y_train = genres_to_vec(category_dict, y_train)
    y_val = genres_to_vec(category_dict, y_val)
    y_test = genres_to_vec(category_dict, y_test)

    return category_dict, (x_train, y_train), (x_val, y_val), (x_test, y_test)

