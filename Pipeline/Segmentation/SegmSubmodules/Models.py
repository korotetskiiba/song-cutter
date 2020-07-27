import numpy as np
from keras_contrib.layers import CRF
from keras.layers import TimeDistributed
from keras.layers import Dense, Dropout
from keras.layers.wrappers import Bidirectional
from keras.layers import Input, GRU
from keras.models import Model
from keras.models import load_model
from keras.utils import to_categorical


SEQ_LEN = 100  # constant


def build_model(embed_dim=128):
    """Builds default baseline model with CRF as the last layer and custom object corresponding to CRF

    Returns:
        keras compiled model"""
    # define model with CRF
    input = Input(shape=(None, embed_dim,))
    model = Bidirectional(GRU(units=64, return_sequences=True,
                                  recurrent_dropout=0.2))(input)  # variational biGRU
    model = Dropout(0.2)(model)
    model = TimeDistributed(Dense(16, activation="relu"))(model)  # a dense layer as suggested by neuralNer
    crf_layer = CRF(2)  # CRF layer, 2 classes (song and not-song)
    out = crf_layer(model)  # output

    model = Model(input, out)
    # CRF layer requires special loss and metrics
    model.compile(optimizer="adam", loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])

    return model


def load_from_cpt(path):
    """Loads keras baseline model with custom objects from checkpoint

    Args:
        path: path to the model checkpoint

    Returns:
        loaded keras model"""
    model = load_model(path, custom_objects=__create_custom_objects())
    return model


def predict_track_pack(x_data, crf_model):
    """Make model prediction on tensor longer than model process with a single iteration. Cut each sample of X tensor
     into equal pieces and predict on them

        Args:
            x_data: tensor of embeddings to make prediction (shape is (samples, duration, embeddings))
            crf_model: keras model with crf as the last layer

        Returns:
            prediction for the whole input tensor (shape is (samples, duration))"""
    single_dim_sample = x_data.shape[1]  # length of each sample
    tracks = int(single_dim_sample / SEQ_LEN)  # count the number of pieces to predict on
    sample_mask = np.zeros((x_data.shape[0], single_dim_sample, 1), dtype=np.float32)  # init tensor for the mask

    for track_num in range(x_data.shape[0]):
        for i in range(tracks):  # make prediction for each piece
            sample = x_data[track_num, i * SEQ_LEN:(i + 1) * SEQ_LEN, :].copy().reshape(1, SEQ_LEN, x_data.shape[2])
            q = crf_model.predict(sample)
            q = q[0, :, 0]
            sample_mask[track_num, i * SEQ_LEN:(i + 1) * SEQ_LEN, 0] = q  # add to the sample mask
        if tracks * SEQ_LEN < single_dim_sample:  # the last piece
            sample = x_data[track_num, tracks * SEQ_LEN:, :].copy().reshape(1, x_data.shape[1], x_data.shape[2])
            q = crf_model.predict(sample)  # predict on the last piece
            sample_mask[track_num, tracks * SEQ_LEN:, 0] = q

    sample_mask = sample_mask.reshape(sample_mask.shape[0], sample_mask.shape[1])  # reshape mask array of tensors
    return sample_mask


def convert_to_crf_format(y_tr, y_val):
    """Translates label tensors from data generator format to CRF-supportable format

    Args:
        y_tr: train labels
        y_val: validation labels

    Returns:
        crf_y_tr, crf_y_val where crf_y_tr is train labels and crf_y_val is validation labels suitable for the model"""
    # convert data for CRF format
    crf_y_tr = [to_categorical(i, num_classes=2) for i in y_tr]
    crf_y_val = [to_categorical(i, num_classes=2) for i in y_val]

    crf_y_tr = np.array(crf_y_tr)
    crf_y_val = np.array(crf_y_val)
    return crf_y_tr, crf_y_val


# private:
def __create_custom_objects():
    """Create custom objects to properly load keras model with CRF as the last layer

    Returns:
        custom objects dictionary which can be instantly passed as custom object parameter to load function"""
    # make some preparation to properly load objects from keras_contribute
    instance_holder = {"instance": None}

    class ClassWrapper(CRF):
        def __init__(self, *args, **kwargs):
            instance_holder["instance"] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)

    def loss(*args):
        method = getattr(instance_holder["instance"], "loss_function")
        return method(*args)

    def accuracy(*args):
        method = getattr(instance_holder["instance"], "accuracy")
        return method(*args)

    return {"ClassWrapper": ClassWrapper, "CRF": ClassWrapper, "crf_loss": loss,
            "crf_viterbi_accuracy": accuracy}


if __name__ == "__main__":
    """This is used to test if the script can build default model"""
    model = build_model()
    model.summary()
