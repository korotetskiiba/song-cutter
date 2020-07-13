import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import pickle
from enum import Enum
import vggish_params as param
import os
# import argparse


class KindOfData(Enum):
    AUDIOSET = 1
    LIVE = 2
    ALL = 3


class DataGenerator:
    data_audioset = None
    data_live = None
    mask_audioset = None
    mask_live = None

    @classmethod
    def get_generated_sample(cls, type: KindOfData, interval: list, path_to_save=None, name=None):
        """
        Get embedding and mask of data

        :param type: type of set to get;
        :param interval: shows which part of the set to get;
        :return: embedding and mask of set to get;
        """
        if type is KindOfData.AUDIOSET:
            # assert cls.data_audioset,'Should generate audioset samples. Use function generate_audioset_sample'
            if cls.data_audioset is None:
                cls.generate_audioset_sample()
            n = len(cls.data_audioset)
            data, mask = cls.data_audioset[int(interval[0] * n): int(interval[1] * n)], cls.mask_audioset[int(interval[0] * n): int(interval[1] * n)]
        elif type is KindOfData.LIVE:
            # assert cls.data_live,'Should generate live samples. Use function generate_live_sample'
            if cls.data_live is None:
                cls.generate_live_sample()
            n = len(cls.data_live)
            data, mask = cls.data_live[int(interval[0] * n): int(interval[1] * n)], cls.mask_live[int(interval[0] * n): int(interval[1] * n)]
        elif type is KindOfData.ALL:
            # assert cls.data_audioset and cls.data_live, 'Should generate audioset and live samples. Use function generate_audioset_sample and generate_live_sample'
            if cls.data_audioset is None:
                cls.generate_audioset_sample()
            if cls.data_live is None:
                cls.generate_live_sample()
            n = len(cls.data_audioset)
            m = len(cls.data_live)

            # cut some piece of data
            data_a = cls.data_audioset[int(interval[0] * n): int(interval[1] * n)]
            mask_a = cls.mask_audioset[int(interval[0] * n): int(interval[1] * n)]
            data_l = cls.data_live[int(interval[0] * m): int(interval[1] * m)]
            mask_l = cls.mask_live[int(interval[0] * m): int(interval[1] * m)]
            data = np.vstack([data_a, data_l])
            mask = np.vstack([mask_a, mask_l])

            # shuffle embeddings and masks of different sets in the same order
            data, mask = cls.__shuffle(data, mask)
        if path_to_save:
            if not os.path.isdir(path_to_save):
                os.mkdir(path_to_save)
            with open(path_to_save +'/'+ name + ".pkl", "wb") as file:
                pickle.dump({'data': data, 'mask': mask}, file)
        return data, mask

    @classmethod
    def generate_audioset_sample(cls, n_samples, path_to_data='bal_train.h5',
                                 path_to_labels='class_labels_indices.csv',
                                 path_to_genres=None):
        """Generate n samples using AudioSet

        :param n_samples: the number of samples to generate;
        :param path_to_data: path to the file containing embedding and samples information;
        :param path_to_labels: path to the file containing information about samples labels;
        :param path_to_genres: path to the file containing additional samples with genres;
        :return: void
        """

        # extract info from audio set
        music_data, speech_data = cls.__extract_info_audioset(path_to_data, path_to_labels)

        # extract info about music with genres classification
        if path_to_genres:
            music_data_new_res = cls.__extract_music_for_genres(path_to_genres)
            # add data to sample
            music_data = np.vstack([music_data, music_data_new_res])

        # generate samples
        cls.data_audioset, cls.mask_audioset = cls.__generate_data_sample(music_data, speech_data, n_samples)

    @classmethod
    def generate_live_sample(cls, path):
        """Generate samples from YouTube video, films etc
                :param path: pkl file containing information about masks and embeddings of samples from live set
                :return: void
        """

        # read data
        with open(path, "rb") as handle:
            data_dict = pickle.load(handle)
        live_embed = cls.__uint8_to_float32([data_dict['embeddings_list']][0])
        mask_list = cls.__bool_to_float32([data_dict['mask_list']][0])

        # shuffle embeddings and masks in the same order
        order = np.random.permutation(mask_list.shape[0])
        cls.data_live = cls.__shuffle(live_embed, order)
        cls.mask_live = cls.__shuffle(mask_list, order)
        # np.zeros((N_samples, seq_len, 1), dtype=np.float32)
        sh = cls.mask_live.shape
        cls.mask_live = cls.mask_live.reshape(sh[0], sh[1], 1)


    @staticmethod
    def __shuffle(arr1, arr2):
        """Shuffles the arrays in the same order
            :param arr1: numpy array to shuffling
            :param arr2: numpy array to shuffling
            :return (new_arr1, new_arr2): arrays shuffled in the same order
        """
        order = np.random.permutation(arr1.shape[0])
        new_arr1 = np.zeros(arr1.shape)
        new_arr2 = np.zeros(arr2.shape)
        for i in range(len(order)):
            new_arr1[order[i]] = arr1[i]
            new_arr2[order[i]] = arr2[i]
        return new_arr1, new_arr2

    @staticmethod
    def __generate_data_sample(music_data, speech_data,
                             n_samples=10000, seq_len=100):
        """Generate random samples by mixing music and speech
            :param music_data: complete numpy array of shape (len, dur, emb), where len is the number of music pieces
            with the duration of dur seconds each with emb-dimensional vector of embeddings
            :param speech_data: the same as music_data but sound pieces correspond "is_speaking" class
            :param n_samples: the number of samples to generate
            :param seq_len: sequence length in seconds, the max length of music part in a single sample
            :return (data_sample, mask_sample) where
                data_sample: numpy tensor of samples with speech and music parts, shape is (sample_index, time, embeddings)
                mask_sample: numpy tensor with masks of the samples with ones in place of music part
        """
        number_tracks = int(round(seq_len) / 10)
        data_sample = np.zeros((n_samples, seq_len, param.EMBEDDING_SIZE), dtype=np.float32)
        mask_sample = np.zeros((n_samples, seq_len, 1), dtype=np.float32)
        for i in tqdm(range(n_samples)):

            # generate audio features
            for k in range(number_tracks):
                rand_idx = np.random.randint(0, len(speech_data))
                data_sample[i, 10 * k:10 * (k + 1), :] = speech_data[rand_idx, :, :]

            # randomly select position for music part
            start = np.random.randint(int(-0.2 * seq_len), int(1.2 * seq_len))
            duration = np.random.randint(seq_len)
            finish = start + duration
            if start < 0:
                start = 0
            if finish < 0:
                continue
            if start < seq_len:
                if finish > seq_len:
                    finish = seq_len
                mask_sample[i, start:finish, 0] = 1.0
                curr_music_len = finish - start
                while curr_music_len > 10:
                    rand_idx = np.random.randint(0, len(music_data))
                    data_sample[i, start:start + 10, :] = music_data[rand_idx, :, :]
                    start += 10
                    curr_music_len = finish - start
                rand_idx = np.random.randint(0, len(music_data))
                data_sample[i, start:finish, :] = music_data[rand_idx, :curr_music_len, :]
        return data_sample, mask_sample

    @classmethod
    def __extract_music_for_genres(cls, path_to_genres):
        """Prepare music of different genres
            :param path_to_genres: path to the file containing additional samples with genres;
            :return music_data_new_res: numpy tensor containing embeddings
        """
        with open(path_to_genres, "rb") as handle:
            data_dict_music = pickle.load(handle)

        # data_dict_music has struct {'category_dict', 'embedings_list', 'label_list'}
        embeddings_list = data_dict_music["embeddings_list"]

        # define some params
        seq_length = embeddings_list[0].shape[0]
        embed_dim = embeddings_list[0].shape[1]
        music_data_new = np.zeros((len(embeddings_list), seq_length, embed_dim), dtype=np.float32)
        for i in range(len(embeddings_list)):
            music_data_new[i, :, :] = embeddings_list[i]

        music_data_new = cls.__uint8_to_float32(music_data_new)
        # reshape 31 frames data to 10 sec frames
        music_data_new_res = music_data_new.reshape(3100, 10, param.EMBEDDING_SIZE)
        return music_data_new_res

    @classmethod
    def __extract_info_audioset(cls, path_to_data, path_to_labels):
        """Prepare samples from AudioSet
            :param path_to_data: path to the file containing embedding and samples information;
            :param path_to_labels: path to the file containing information about samples labels;
            :return (music_data, speech_data): where
                music_data: complete numpy array of shape (len, dur, emb), where len is the number of
                 pieces with the duration of dur seconds each with emb-dimensional vector of embeddings
                speech_data: the same as music_data but sound pieces correspond "is_speaking" class
        """

        # load info about samples
        (x, y, video_id_list) = cls.__load_data(path_to_data)
        x = cls.__uint8_to_float32(x)  # shape: (N, 10, 128)
        y = cls.__bool_to_float32(y)  # shape: (N, 527)

        # find speech and sing labels
        labels = pd.read_csv(path_to_labels)
        speech_labels = labels[labels["is_speaking"] == 1]["index"].values
        music_labels = labels[labels["is_singing"] == 1]["index"].values

        # collect all speech records
        # recognize indexes of samples with music or speech
        music_idxs = []
        speech_idxs = []
        for i, val in tqdm(enumerate(y)):
            nnz_idxs = val.nonzero()[0]
            if any([idx in music_labels for idx in nnz_idxs]):
                music_idxs.append(i)
            if any([idx in speech_labels for idx in nnz_idxs]):
                speech_idxs.append(i)

        # do sets of indexes are disjoint
        music_set = set(music_idxs)
        speech_set = set(speech_idxs)
        music_set_clear = music_set.difference(speech_set)
        speech_set_clear = speech_set.difference(music_set)

        # collect arrays for speech and singing
        # music_data, speech_data is tensor of embeddings
        music_data = np.zeros((len(music_set_clear), 10, param.EMBEDDING_SIZE), dtype=np.float32)
        speech_data = np.zeros((len(speech_set_clear), 10, param.EMBEDDING_SIZE), dtype=np.float32)
        for i, idx in enumerate(music_set_clear):
            music_data[i, :, :] = x[idx, :, :]
        for i, idx in enumerate(speech_set_clear):
            speech_data[i, :, :] = x[idx, :, :]
        return music_data, speech_data

    @staticmethod
    def __load_data(hdf5_path):
        with h5py.File(hdf5_path, 'r') as hf:
            x = hf.get('x')
            y = hf.get('y')
            video_id_list = hf.get('video_id_list')
            x = np.array(x)
            y = list(y)
            video_id_list = list(video_id_list)
        return x, y, video_id_list

    @staticmethod
    def __uint8_to_float32(x):
        return (np.float32(x) - 128.) / 128.

    @staticmethod
    def __bool_to_float32(y):
        return np.float32(y)

if __name__ == "__main__":
    pass