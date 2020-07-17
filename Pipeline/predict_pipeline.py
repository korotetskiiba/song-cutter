from Pipeline.Preprocessing.PreprocessingModule import PreprocessingModule as pp
from Pipeline.FeatureExtractor.FeatureExtraction import FeatureExtraction as fe
from Pipeline.Segmentation.SegmentationModule import SegmentationModule as sg
import os
import pickle
import numpy as np

# define some path
# for preprocessing
PATH_TO_VIDEO_FOLDER = os.path.join('auxiliary_files', 'video_file_for_predict')
# for feature extraction
PATH_TO_WAV_FOLDER = os.path.join('auxiliary_files', 'wav_files_for_predict')
PATH_TO_DATA_WITH_EMBEDDINGS = os.path.join('auxiliary_files', 'emb_for_predict.pkl')
# for model
PATH_TO_CHECKPOINT_FILE = os.path.join('auxiliary_files', 'model.h5')


def uint8_to_float32(x):
    return (np.float32(x) - 128.) / 128.

if __name__ == "__main__":
    pp.convert_to_wav(PATH_TO_VIDEO_FOLDER, PATH_TO_WAV_FOLDER)
    fe.get_audioset_features_file(PATH_TO_WAV_FOLDER, PATH_TO_DATA_WITH_EMBEDDINGS)

    with open(PATH_TO_DATA_WITH_EMBEDDINGS, "rb") as handle:
        data_dict_music = pickle.load(handle)
    files_list = data_dict_music["files_list"]
    test_sample = data_dict_music["embeddings_list"]
    # if len(files_list) == 1:
    #     test_sample = uint8_to_float32(data_dict_music["embeddings_list"])
    # else:
    #     test_sample = uint8_to_float32(data_dict_music["embeddings_list"][0])

    model = sg()
    model.load_from_checkpoint(PATH_TO_CHECKPOINT_FILE)
    for i in range(len(files_list)):
        sample = uint8_to_float32(test_sample[i])
        sample = sample.reshape([1, sample.shape[0], sample.shape[1]])
        intervals = model.predict(sample)
        name = os.path.basename(files_list[i])[:-3]
        path = os.path.join(PATH_TO_VIDEO_FOLDER, name + 'mp4')
        model.cut_file(path, PATH_TO_VIDEO_FOLDER, intervals)