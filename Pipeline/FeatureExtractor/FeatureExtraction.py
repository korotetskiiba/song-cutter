import pickle
from Pipeline.FeatureExtractor.vggish_files import VGGishModel


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
    import sys
    assert len(sys.argv) >= 3, "wrong num of arguments"
    FeatureExtraction.get_audioset_features(sys.argv[1], sys.argv[2])
