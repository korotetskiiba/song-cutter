from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

from Pipeline.MetaExtraction.MetaExtrSubmodules import Models, Evaluation
from keras.models import load_model
import numpy as np


class GenreClassifier:
    """
    This class represents pipeline module responsible for genre classification.
    """
    def __init__(self):
        """
        Initializes module.
        """
        self.model = None
        self.category_dict = None

    def load_from_checkpoint(self, path_to_checkpoint_file,
                             genre_dict=None):
        """
        Loads trained model into self.model from checkpoint file.

        :param path_to_checkpoint_file: path to the checkpoint(.h5) file;
        :param genre_dict: dict{<number>:<genre>};
        :return: void.
        """
        assert isinstance(path_to_checkpoint_file, str) and path_to_checkpoint_file.endswith('.h5'),\
            "Wrong checkpoint file argument!"
        self.model = load_model(path_to_checkpoint_file)
        # default category dict
        if genre_dict is None:
            self.category_dict = {0: 'none', 1: 'pop', 2: 'rock',
                                  3: 'jazz', 4: 'reggae', 5: 'metal',
                                  6: 'country', 7: 'indi', 8: 'electronic',
                                  9: 'hiphop', 10: 'disco', 11: 'blues',
                                  12: 'folk', 13: 'classical', 14: 'experimental',
                                  15: 'instrumental', 16: 'international'
                                  }
        else:
            self.category_dict = genre_dict

    def exec_fit(self, x_train, x_valid, y_train, y_valid,
                 checkpoint_file, category_dict, type="RNN", epochs=300, batch_size=16):
        """
        Given train and validation sets, builds a model(RNN or CNN) following argument 'type', executes fit and
        saves trained model as checkpoint file at given path.

        :param x_train: train data tensor (number of samples, duration, embedding) (num, 31, 128);
        :param x_valid: validation data tensor;
        :param y_train: train data labels tensor;
        :param y_valid: validation data labels tensor;
        :param checkpoint_file: name of file where checkpoints will be saved;
        :param category_dict: dictionary {<genre number>: <genre name>};
        :param type: what type of NN to choose if model not loaded yet("CNN" or "RNN");
        :param epochs: the number of epochs to fit;
        :param batch_size: the number of samples in a single batch to fit;
        :return: model saved at given path.
        """
        if self.category_dict is None:
            self.category_dict = category_dict
        callback_list = self.__define_callback_list(checkpoint_file)
        # build default model
        if self.model is None:
            self.__build_new_model(type=type)
        self.model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_valid, y_valid),
                       epochs=epochs, callbacks=callback_list)

    def predict(self, x_data):
        """
        Given a list of samples in form of second-wise embeddings returns a list of labels predicted for each sample.

        :param x_data: train data tensor (number of samples, duration, embedding) (num, 31, 128);
        :return: labels for samples.
        """
        y_pred = self.model.predict(x_data)
        pred_labels_nums = [np.argmax(v) for v in y_pred]
        pred_labels = [self.category_dict[i] for i in pred_labels_nums]
        return pred_labels

    def __build_new_model(self, type="RNN", embed_dim=128):
        """
        Builds classifier model according to given type and embedding dimensions.

        :param type: 'RNN' or 'CNN';
        :param embed_dim: dimension of the embedding used;
        :return: void. self.model becomes initialized after calling this method.
        """
        if type == "RNN":
            self.model = Models.build_RNN_model(num_of_classes=len(self.category_dict), embed_dim=embed_dim)
        if type == "CNN":
            self.model = Models.build_CNN_model(num_of_classes=len(self.category_dict), embed_dim=embed_dim)

    def evaluate(self, pred_labels, y_test, target_path):
        """
        Calculates F1 and accuracy. Plots and saves confusion matrices as png files.

        :param pred_labels: labels predicted by the model;
        :param y_test: ground truth labels;
        :param target_path: path to save artefacts;
        :return: void.
        """
        # reformat label-info from vecs to words
        target_labels = [self.category_dict[np.nonzero(mask)[0][0]] for mask in y_test]
        Evaluation.count_metrics_on_sample(pred_labels, target_labels)
        Evaluation.cnf_matrices(pred_labels, target_labels, self.category_dict, target_path)

    @staticmethod
    def __define_callback_list(checkpoint_file, custom_callback_list=None):
        """
        Get callbacks list to pass it to the fit. Callbacks include learning rate scheduler, checkpoint and
        early stopping.

        :param checkpoint_file: path to file where model checkpoint will be saved;
        :param custom_callback_list: custom callbacks to use in model fit. If None, no extra custom callbacks added;
        :return: the list of callbacks which can be passed to fit as callback parameter instantly.
        """
        # define learning rate schedule
        lr_scheduler_callback = LearningRateScheduler(lambda epoch: 0.003 if epoch < 5 else 0.001)

        # define checkpoints
        cpt_callback = ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=0, save_best_only=True,
                                       save_weights_only=False, mode='auto')

        # define early stopping
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=7, verbose=0)

        result_cb_list = [cpt_callback, lr_scheduler_callback, early_stopping_callback]

        # add custom callbacks
        if custom_callback_list is not None:
            result_cb_list.extend(custom_callback_list)

        return result_cb_list
