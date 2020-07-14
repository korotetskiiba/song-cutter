import pickle
from Pipeline.FeatureExtractor.vggish_files import VGGishModel
import argparse


class FeatureExtraction:

    @staticmethod
    def get_audioset_features(path_to_pkl, path_to_save_pkl):

        assert isinstance(path_to_pkl, str) and path_to_pkl.endswith('.pkl'), "Wrong path to pkl argument"
        assert isinstance(path_to_save_pkl, str) and path_to_save_pkl.endswith('.pkl'), "Wrong path to save pkl argument"

        with open(path_to_pkl, "rb") as handle:
            data_dict_music = pickle.load(handle)

        #reading a list of files
        file_list = data_dict_music.get("files_list")
        assert file_list is not None, "files list not found in "+path_to_pkl

        #model definition
        model = VGGishModel.VGGishModel()

        data_dict_music["embeddings_list"] = model.get_embeddings_list(file_list)

        with open(path_to_save_pkl, "wb") as handle:
            pickle.dump(data_dict_music, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing')

    parser.add_argument('-i', type=str, action="store", dest="path_to_pkl", help='path to input file')
    parser.add_argument('-o', type=str, action="store", dest="path_to_save_pkl",
                        help='path to save the output file')
    args = parser.parse_args()

    FeatureExtraction.get_audioset_features(args.path_to_pkl, args.path_to_save_pkl)
