# from Pipeline.Preprocessing.PreprocessingModule import PreprocessingModule as pp
# from Pipeline.FeatureExtractor.FeatureExtraction import FeatureExtraction as fe
# from Pipeline.Segmentation.SegmentationModule import SegmentationModule as sg
import os

# for preprocessing
PATH_TO_VIDEO_FOLDER = os.path.join('auxiliary_files', 'video_files')
PATH_TO_WAV_FOLDER = os.path.join('auxiliary_files', 'wav_files_for_predict')
# for feature extraction
PATH_TO_DATA = os.path.join('auxiliary_files', 'wav_files', 'pickle_of_this_folder.pkl')
PATH_TO_DATA_WITH_EMBEDDINGS = os.path.join('auxiliary_files', 'res.pkl')
# for model
PATH_TO_CHECKPOINT_FILE = os.path.join('auxiliary_files', 'model.h5')

import os
import pickle

if __name__ == "__main__":
    pass
    # pp.convert_to_wav(PATH_TO_VIDEO_FOLDER, PATH_TO_WAV_FOLDER)
    # file_list = os.listdir(PATH_TO_WAV_FOLDER)
    # file_list = [PATH_TO_WAV_FOLDER +'/' + file for file in file_list ]
    # data = {'files_list': file_list}
    # with open(PATH_TO_DATA_WITH_EMBEDDINGS, "rb") as handle:
    #     data_dict_music = pickle.load(handle)
    # print(data_dict_music)

    # path_to_pkl = PATH_TO_DATA
    # with open(path_to_pkl, "rb") as handle:
    #     data_dict_music = pickle.load(handle)
    # print(data_dict_music)

    # fe.get_audioset_features(PATH_TO_DATA, PATH_TO_DATA_WITH_EMBEDDINGS)
    # model = sg()
    # model.exec_fit(x_tr, x_val, y_tr, y_val, PATH_TO_CHECKPOINT_FILE, epochs=5, batch_size=5)
    # model.load_from_checkpoint(PATH_TO_CHECKPOINT_FILE)
    # model.evaluate(x_te, y_te, 'wav_files')