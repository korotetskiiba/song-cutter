import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import pickle
from enum import Enum


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
    def get_generate(cls, type: KindOfData, interval: list):
        """Get embedding and mask of data"""
        if type is KindOfData.AUDIOSET:
            n = len(cls.data_audioset)
            return cls.data_audioset[int(interval[0] * n): int(interval[1] * n)], cls.mask_audioset[int(interval[0] * n): int(interval[1] * n)]
        if type is KindOfData.LIVE:
            n = len(cls.data_live)
            return cls.data_live[int(interval[0] * n): int(interval[1] * n)], cls.mask_live[int(interval[0] * n): int(interval[1] * n)]
        if type is KindOfData.ALL:
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
            order = np.random.permutation(mask.shape[0])
            data = cls.__shuffle(data, order)
            mask = cls.__shuffle(mask, order)
            return data, mask

    @classmethod
    def generate_audioset_sample(cls, n_samples, path_to_data='bal_train.h5',
                                 path_to_labels='class_labels_indices.csv',
                                 path_to_genres=None):
        """Generate n samples using AudioSet"""

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
        """Generate samples from YouTube video, films etc"""

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
    def __shuffle(arr, order):
        new_arr = np.zeros(arr.shape)
        for i in range(len(order)):
            new_arr[order[i]] = arr[i]
        return new_arr

    @staticmethod
    def __generate_data_sample(music_data, speech_data,
                             n_samples=10000, seq_len=100, embed_dim=128):
        # Generate random samples by mixing music and speech
        number_tracks = int(round(seq_len) / 10)
        data_sample = np.zeros((n_samples, seq_len, embed_dim), dtype=np.float32)
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
        # prepare music of different genres
        with open(path_to_genres, "rb") as handle:
            data_dict_music = pickle.load(handle)

        # data_dict_music has struct {'category_dict', 'embedings_list', 'label_list'}
        # category_dict - dictionary with genres: blues, classical, etc
        # embeddings_list - list with embeddings of samples. Every sample has the same len
        # label_list - lables this sample by category
        embeddings_list = data_dict_music["embeddings_list"]

        # define some params
        seq_length = embeddings_list[0].shape[0]
        embed_dim = embeddings_list[0].shape[1]
        music_data_new = np.zeros((len(embeddings_list), seq_length, embed_dim), dtype=np.float32)
        for i in range(len(embeddings_list)):
            music_data_new[i, :, :] = embeddings_list[i]

        music_data_new = cls.__uint8_to_float32(music_data_new)
        # reshape 31 frames data to 10 sec frames
        music_data_new_res = music_data_new.reshape(3100, 10, 128)
        return music_data_new_res

    @classmethod
    def __extract_info_audioset(cls, path_to_data, path_to_labels):
        # load info about samples
        # for every samples:
        # x - embeddings, y - info about labels, video_id_list - id of this sample
        # y is a array len of num_of_labels with 0 and 1, where 1 in i place means sample belongs to i label, 0 -doesn't
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
        music_data = np.zeros((len(music_set_clear), 10, 128), dtype=np.float32)
        speech_data = np.zeros((len(speech_set_clear), 10, 128), dtype=np.float32)
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

DataGenerator.generate_audioset_sample(7)
DataGenerator.generate_live_sample('music_new.pkl')
print(DataGenerator.data_live.shape, DataGenerator.mask_live.shape)
print(DataGenerator.data_audioset.shape, DataGenerator.mask_audioset.shape)

# print('\n\n\n')
# print(DataGenerator.data_live, DataGenerator.mask_live)
# print(DataGenerator.data_audioset, DataGenerator.mask_audioset)
print(DataGenerator.get_generate(KindOfData.ALL, [0, 0.4]))