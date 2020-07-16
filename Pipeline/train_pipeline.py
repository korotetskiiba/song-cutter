from Pipeline.Preprocessing.PreprocessingModule import PreprocessingModule as pp
from Pipeline.FeatureExtractor.FeatureExtraction import FeatureExtraction as fe
from Pipeline.DataGenerator.DataGenerator import DataGenerator as dg
from Pipeline.DataGenerator.DataGenerator import KindOfData as kd
from Pipeline.Segmentation.SegmentationModule import SegmentationModule as sg

# define some path
PATH_TO_META_FOLDER = 'auxiliary_files/data_to_preproc'
PATH_TO_VIDEO_FOLDER = 'auxiliary_files/video_files'
PATH_TO_WAV_FOLDER = 'auxiliary_files/wav_files'
PATH_TO_LIVE_DATA_WITH_EMBEDDINGS = 'auxiliary_files/res.pkl'
PATH_TO_LIVE_DATA = 'auxiliary_files/wav_files/pickle_samples.pkl' 
PATH_TO_CHECKPOINT_FILE = 'auxiliary_files/model.h5'
PATH_TO_METRIC_WITHOUT_TR = 'auxiliary_files/metrics/metrics_without_additional_tr'
PATH_TO_METRIC_WITH_TR = 'auxiliary_files/metrics/metrics_with_additional_tr'

if __name__ == "__main__":
    # preprocesing live samples ang get embeddings
    pp.preprocess_train(PATH_TO_META_FOLDER, PATH_TO_VIDEO_FOLDER, PATH_TO_WAV_FOLDER, seq_len=96)
    fe.get_audioset_features(PATH_TO_LIVE_DATA, PATH_TO_LIVE_DATA_WITH_EMBEDDINGS)

    # generate audioset and liveset samples
    sets = dg.get_generated_sample(kd.AUDIOSET, [7, 3, 1])
    x_tr, y_tr = sets['train']
    x_val, y_val = sets['val']
    x_test, y_test = sets['test']

    sets_l = dg.get_generated_sample(kd.LIVE, [7, 3, 1], path_to_live_data=PATH_TO_LIVE_DATA_WITH_EMBEDDINGS)
    x_tr_l, y_tr_l = sets_l['train']
    x_val_l, y_val_l = sets_l['val']
    x_test_l, y_test_l = sets_l['test']

    # create model
    model = sg()
    # train model and evaluate
    model.exec_fit(x_tr, x_val, y_tr, y_val, PATH_TO_CHECKPOINT_FILE)
    model.evaluate(x_test_l, y_test_l, PATH_TO_METRIC_WITHOUT_TR)
    # additional train and evaluate
    model.exec_fit(x_tr_l, x_val_l, y_tr_l, y_val_l, PATH_TO_CHECKPOINT_FILE)
    model.evaluate(x_test_l, y_test_l, PATH_TO_METRIC_WITH_TR)