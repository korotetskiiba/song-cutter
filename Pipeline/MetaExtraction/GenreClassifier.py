from sklearn.metrics import log_loss, f1_score, classification_report, confusion_matrix, accuracy_score
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from Pipeline.MetaExtraction.MetaExtrSubmodules import Models
from sklearn.model_selection import train_test_split
from keras.models import load_model
import numpy as np
import joblib


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

    def load_from_checkpoint(self, path_to_checkpoint_file):
        """
        Loads trained model into self.model from checkpoint file.

        :param path_to_checkpoint_file: path to the checkpoint(.h5) file;
        :return: void.
        """
        assert isinstance(path_to_checkpoint_file, str) and path_to_checkpoint_file.endswith('.h5'),\
            "Wrong checkpoint file argument!"
        self.model = load_model(path_to_checkpoint_file)

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
        if self.model is None:  # build default model
            self.__build_new_model(type=type)
        self.model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_valid, y_valid),
                       epochs=epochs, callbacks=callback_list)

    def predict(self, x_data):
        """
        Given a list of samples in form of second-wise embeddings returns a list of labels predicted for each sample.

        :param x_data: train data tensor (number of samples, duration, embedding) (num, 31, 128)
        :return: labels for samples
        """
        y_pred = self.model.predict(x_data)
        pred_label = [np.argmax(v) for v in y_pred]
        return pred_label

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


if __name__ == "__main__":
    data_dict = joblib.load("D:\current_work\!!!KURS\EPAM\song-cutter\stuff\genres_dump.pkl")
    labels = data_dict["label_list"]
    embedings_list = data_dict["embedings_list"]
    category_dict = data_dict["category_dict"]
    inv_category_dict = {category_dict[k]: k for k in category_dict}

    seq_length = embedings_list[0].shape[0]
    embed_dim = embedings_list[0].shape[1]
    nb_classes = len(category_dict)

    target = np.zeros((len(labels), len(category_dict)), dtype=np.float32)
    data = np.zeros((len(embedings_list), seq_length, embed_dim), dtype=np.float32)

    for i in range(len(target)):
        target[i, inv_category_dict[labels[i]]] = 1.0
        data[i, :, :] = embedings_list[i]

    labels = [inv_category_dict[k] for k in labels]

    tr_data, te_data, y_tr, y_te = train_test_split(data, target, test_size=0.2, stratify=labels)
    tr_data, val_data, y_tr, y_val = train_test_split(tr_data, y_tr, test_size=0.2)

    filepath = 'D:\current_work\!!!KURS\EPAM\song-cutter\model_cpt\gru_64_avg_pool_gaus_do_batch_16.h5'
    gc = GenreClassifier()
    # gc.exec_fit(tr_data, val_data, y_tr, y_val, filepath, type="RNN")
    gc.load_from_checkpoint(filepath)
    pred_label = gc.predict(te_data)
    true_label = [np.argmax(v) for v in y_te]
    print(pred_label)
    print(true_label)
    print(f1_score(true_label, pred_label, average="micro"))
