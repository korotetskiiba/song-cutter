from Pipeline.Preprocessing.PreprocessingModule import PreprocessingModule as pp
from Pipeline.FeatureExtractor.FeatureExtraction import FeatureExtraction as fe
from Pipeline.DataGenerator.DataGenerator import DataGenerator as dg
from Pipeline.DataGenerator.DataGenerator import KindOfData as kd
from Pipeline.Segmentation.SegmentationModule import SegmentationModule as sg

PATH_TO_META_FOLDER = 'auxiliary_files/data_to_preproc'
PATH_TO_VIDEO_FOLDER = 'auxiliary_files/video_files'
PATH_TO_WAV_FOLDER = 'auxiliary_files/wav_files'
PATH_TO_LIVE_DATA_WITH_EMBEDDINGS = 'auxiliary_files/res.pkl'
PATH_TO_LIVE_DATA = 'auxiliary_files/wav_files/pickle_of_this_folder.pkl'
PATH_TO_CHECKPOINT_FILE = 'auxiliary_files/model.h5'

if __name__ == "__main__":
    pp.preprocess_train(PATH_TO_META_FOLDER, PATH_TO_VIDEO_FOLDER, PATH_TO_WAV_FOLDER, seq_len=96)
    fe.get_audioset_features(PATH_TO_LIVE_DATA, PATH_TO_LIVE_DATA_WITH_EMBEDDINGS)
    sets = dg.get_generated_sample(kd.AUDIOSET, [0.7, 0.3, 0.1])
    x_tr, y_tr = sets['train']
    x_val, y_val = sets['val']
    x_test, y_test = sets['test']

    # sets = dg.get_generated_sample(kd.ALL, [70,20,10], path_to_live_data=PATH_TO_LIVE_DATA_WITH_EMBEDDINGS)
    # x = x_test[0].copy()
    # y = y_test[0].copy()
    # x = x.reshape(1, x.shape[0], x.shape[1])
    # y=y.reshape(-1)

    model = sg()
    model.exec_fit(x_tr, x_val, y_tr, y_val, PATH_TO_CHECKPOINT_FILE)
    # sets = dg.get_generated_sample(kd.LIVE, [700, 300, 5], path_to_live_data=PATH_TO_LIVE_DATA_WITH_EMBEDDINGS)
    # x_tr, y_tr = sets['train']
    # x_val, y_val = sets['val']
    # x_test, y_test = sets['test']
    # # model.evaluate(x, y, 'auxiliary_files/wav_files')
    # model.exec_fit(x_tr, x_val, y_tr, y_val, PATH_TO_CHECKPOINT_FILE, epochs=5, batch_size=5)
    # model.evaluate(x, y, 'auxiliary_files/wav_files')