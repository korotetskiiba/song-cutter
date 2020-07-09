import numpy as np
from keras_contrib.layers import CRF
from keras.layers import TimeDistributed
from keras.layers import Dense, Dropout
from keras.layers.wrappers import Bidirectional
from keras.layers import Input, GRU
from keras.models import Model
from keras.models import load_model


SEQ_LEN = 100  # constant


def build_model():
    # define model with CRF
    embed_dim = 128

    input = Input(shape=(None, embed_dim,))
    model = Bidirectional(GRU(units=64, return_sequences=True,
                                  recurrent_dropout=0.2))(input)  # variational biGRU
    model = Dropout(0.2)(model)
    model = TimeDistributed(Dense(16, activation="relu"))(model)  # a dense layer as suggested by neuralNer
    crf_layer = CRF(2)  # CRF layer
    out = crf_layer(model)  # output

    model = Model(input, out)
    model.compile(optimizer="adam", loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])

    return model


def load_from_cpt(path):
    model = load_model(path, custom_objects=__create_custom_objects())
    return model


def predict_mask_long_sample(x_data, crf_model):
    single_dim_sample = x_data.shape[1]
    tracks = int(single_dim_sample / SEQ_LEN)
    sample_mask = np.zeros((single_dim_sample, 1), dtype=np.float32)

    for i in range(tracks):
        q = crf_model.predict(x_data[:, i * SEQ_LEN:(i + 1) * SEQ_LEN, :])
        q = q[0, :, 0]
        sample_mask[i * SEQ_LEN:(i + 1) * SEQ_LEN, 0] = q

    q = crf_model.predict(sample_mask[:, tracks * SEQ_LEN:, :])
    q = q[0, :, 0]
    sample_mask[tracks * SEQ_LEN:, 0] = q
    sample_mask = sample_mask.reshape(-1)
    return sample_mask


# private:
def __create_custom_objects():
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
    model = build_model()
    model.summary()
