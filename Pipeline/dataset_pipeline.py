from Pipeline.Preprocessing.PreprocessingModule import PreprocessingModule as pp
from Pipeline.FeatureExtractor.FeatureExtraction import FeatureExtraction as fe
from Pipeline.DataGenerator.DataGenerator import DataGenerator as dg
from Pipeline.DataGenerator.DataGenerator import KindOfData as kd

import os
import pickle


# define some path
# for preprocessing
PATH_TO_META_FOLDER = os.path.join('auxiliary_files', 'data_to_preproc')
PATH_TO_VIDEO_FOLDER = os.path.join('auxiliary_files', 'video_files')
PATH_TO_WAV_FOLDER = os.path.join('auxiliary_files', 'wav_files')
# for feature extraction
PATH_TO_LIVE_DATA = os.path.join('auxiliary_files', 'wav_files')
LIVE_DATA_TARGET = 'pickle_samples.pkl'
# for feature extraction and data generator
PATH_TO_LIVE_DATA_WITH_EMBEDDINGS = os.path.join('auxiliary_files', 'products')
LIVE_WITH_EMBEDDINGS_TARGET = 'res.pkl'
# for dataset
PATH_TO_DATASET = os.path.join('auxiliary_files', 'dataset')
DATASET_NAME = 'live_set.pkl'


if __name__ == "__main__":

    SEQ_LEN = 96
    dataset_dict = {}
    for data_part in ['train', 'valid', 'test']:
        # manage paths
        path_to_meta = os.path.join(PATH_TO_META_FOLDER, data_part)
        path_to_video = os.path.join(PATH_TO_VIDEO_FOLDER, data_part)
        path_to_wav = os.path.join(PATH_TO_WAV_FOLDER, data_part)
        path_to_live_data_dir = os.path.join(PATH_TO_LIVE_DATA, data_part)
        path_to_live_data_with_embeddings_dir = os.path.join(PATH_TO_LIVE_DATA_WITH_EMBEDDINGS, data_part)

        # ensure dirs exist
        for cur_path in [path_to_meta, path_to_video, path_to_wav, path_to_live_data_dir,
                         path_to_live_data_with_embeddings_dir]:
            if not os.path.isdir(cur_path):
                os.makedirs(cur_path, exist_ok=True)

        # create paths to the target pickles
        path_to_live_data = os.path.join(path_to_live_data_dir, LIVE_DATA_TARGET)
        path_to_live_data_with_embeddings = os.path.join(path_to_live_data_with_embeddings_dir,
                                                         LIVE_WITH_EMBEDDINGS_TARGET)

        # preprocessing live samples and get embeddings
        pp.preprocess_train(path_to_meta, path_to_video, path_to_wav, seq_len=SEQ_LEN)
        fe.get_audioset_features(path_to_live_data, path_to_live_data_with_embeddings)

        # generate liveset samples
        sets = dg.get_generated_sample(kd.LIVE, [1, 0, 0], path_to_live_data=path_to_live_data_with_embeddings,
                                       need_shuffle=False)

        x, y = sets['train']  # get samples
        dataset_dict[data_part] = (x, y)

    # save dataset
    if not os.path.isdir(PATH_TO_DATASET):
        os.makedirs(PATH_TO_DATASET, exist_ok=True)

    dataset_filepath = os.path.join(PATH_TO_DATASET, DATASET_NAME)
    with open(dataset_filepath, "wb") as f:
        pickle.dump(dataset_dict, f)

