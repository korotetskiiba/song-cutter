import argparse
import pickle


def run_from_cmd(segmentation_module):
    """Parse args from command line and launch module according to the command line arguments

    Command line parameters:
        -i: path_to_pkl file with numpy tensors packed the same as DataGenerator-output
        -c: checkpoint_file with *.h5 extension where segmentation model will be saved or from where model will be
         loaded
        -o: path_to_file (video or sound) to be cut into slices with music
        -t: target_path where all artifacts will be saved (metrics or slices)
        -f: func, scenario of module work
            values of -f:
                fit: launches fit
                load_predict: load model from checkpoint and make prediction
                load_evaluate: load model from checkpoint and evaluate it
                load_cut: load model from checkpoint and slice video

    Args:
         segmentation_module: SegmentationModule instance
    """
    parser = argparse.ArgumentParser(description='Segmentation', add_help=False)

    parser.add_argument('-i', action="store", dest="path_to_pkl", help="*.pkl file with data tensors")
    parser.add_argument('-c', action="store", dest="checkpoint_file", help="*.h5 file with model checkpoint")
    parser.add_argument('-o', action="store", dest="path_to_file", help="path to video to be cut")
    parser.add_argument('-t', action="store", dest="target_path", help="path to the target directory")
    parser.add_argument('-f', action="store", dest="func", help="function to launch."
                                                               "fit - to launch fit."
                                                               "load_predict - to load checkpoint and make prediction."
                                                               "load_evaluate - to load checkpoint and evaluate model."
                                                               "load_cut - to load checkpoint and slice video.")

    args = parser.parse_args()

    # choose scenario
    if args.func == "fit":
        __fit(segmentation_module=segmentation_module, path_to_pkl=args.path_to_pkl,
              checkpoint_file=args.checkpoint_file, target_path=args.target_path)
    elif args.func == "load_predict":
        prediction_intervals = __load_predict(segmentation_module=segmentation_module,
                                              path_to_pkl=args.path_to_pkl,
                                              checkpoint_file=args.checkpoint_file)
        print("The prediction intervals are:")
        print(prediction_intervals)
    elif args.func == "load_evaluate":
        __load_evaluate(segmentation_module=segmentation_module,
                        path_to_pkl=args.path_to_pkl, checkpoint_file=args.checkpoint_file,
                        target_path=args.target_path)
    elif args.func == "load_cut":
        __load_cut(segmentation_module=segmentation_module, path_to_pkl=args.path_to_pkl,
                   checkpoint_file=args.checkpoint_file, path_to_file=args.path_to_file, target_path=args.target_path)
    else:
        print("Wrong func argument")


def __fit(segmentation_module, path_to_pkl, checkpoint_file, target_path):
    """Executes fit scenario (fit model and evaluate it)

    Args:
        segmentation_module: SegmentationModule instance
        path_to_pkl: path to file with data tensors
        checkpoint_file: path to save model checkpoints
        target_path: path to save evaluation summary (plots and metrics)
    """
    ((x_train, y_train), (x_val, y_val), (x_test, y_test)) = __tensors_from_pkl(path_to_pkl)
    segmentation_module.exec_fit(x_train, x_val, y_train, y_val, checkpoint_file)
    segmentation_module.evaluate(x_test, y_test, target_path)


def __load_predict(segmentation_module, path_to_pkl, checkpoint_file):
    """Executes predict scenario (load model and make prediction)

    Args:
        segmentation_module: SegmentationModule instance
        path_to_pkl: path to file with data tensors
        checkpoint_file: path with model checkpoint to load model
    """
    ((_, _), (_, _), (x_test, _)) = __tensors_from_pkl(path_to_pkl)
    segmentation_module.load_from_checkpoint(checkpoint_file)
    prediction_intervals = segmentation_module.predict(x_test)
    return prediction_intervals


def __load_evaluate(segmentation_module, path_to_pkl, checkpoint_file, target_path):
    """Executes evaluate scenario (load model and evaluate it)

    Args:
        segmentation_module: SegmentationModule instance
        path_to_pkl: path to file with data tensors
        checkpoint_file: path with model checkpoint to load model
        target_path: path to save evaluation summary (plots and metrics)
    """
    ((_, _), (_, _), (x_test, y_test)) = __tensors_from_pkl(path_to_pkl)
    segmentation_module.load_from_checkpoint(checkpoint_file)
    segmentation_module.evaluate(x_test, y_test, target_path)


def __load_cut(segmentation_module, path_to_pkl, checkpoint_file, path_to_file, target_path):
    """Executes cut scenario (load model and cut video or sound into slices with music)

    Args:
        segmentation_module: SegmentationModule instance
        path_to_pkl: path to file with data tensors
        checkpoint_file: path with model checkpoint to load model
        target_path: path to save slices
    """
    time_intervals = __load_predict(segmentation_module, path_to_pkl, checkpoint_file)
    segmentation_module.cut_file(path_to_file, target_path, time_intervals)


def __tensors_from_pkl(path_to_file):
    """Loads tensors with train, validation and test data from file generated by DataGenerator

    Args:
        path_to_file: *.pkl file with data tensors

    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test) where:
            x_train: train data tensor
            y_train: train data mask
            x_val: validation data tensor
            y_val: validation data mask
            x_test: test data tensor
            y_test: test data mask
    """
    with open(path_to_file, "rb") as f:
        data_dict = pickle.load(f)

    return data_dict["train"], data_dict["val"], data_dict["test"]
