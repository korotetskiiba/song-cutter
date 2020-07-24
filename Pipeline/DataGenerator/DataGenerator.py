import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import pickle
from enum import Enum
import Pipeline.FeatureExtractor.vggish_files.vggish_params as param
import os
import argparse


class KindOfData(Enum):
    AUDIOSET = 1
    LIVE = 2
    ALL = 3


class DataGenerator:
    #for live sets
    __genre_list = None
    __data_live = None
    __mask_live = None
    #for audio sets
    __music_data = None
    __speech_data = None
    __path_to_audioset_data = 'DataGenerator/meta_files/bal_train.h5'
    __path_to_labels = 'DataGenerator./meta_files/class_labels_indices.csv'
    __path_to_genres = 'DataGenerator./meta_files/genres_dump.pkl'

    @classmethod
    def get_generated_sample(cls, type: KindOfData, relation: list, n_samples:int=10000, path_to_live_data=None, path_to_save=None, name='sets',
                             need_shuffle:bool=True):
        """
        Get embedding and mask of data
        :param type: type of set to get;
        :param relation: relation for the capacities of sets (train:val:test);
        :param n_samples: the number of samples to generate;
        :param path_to_live_data: pkl file containing information about masks and embeddings of samples from live set
        :param path_to_save: path to folder for saving samples;
        :param name: name of file to save samples
        :param need_shuffle: flag, False if it's not needed to shuffle data;
        :return: result: a dictionary containing train, validation and test samples
        """

        # prepare coefficients
        s = sum(relation)
        k = [relation[0] / s]
        k.append((relation[0] + relation[1]) / s)
        k.append(1)

        # choose mode
        if type is KindOfData.AUDIOSET:
            result = cls.__get_generated_audioset_samples(n_samples, k)
        elif type is KindOfData.LIVE:
            result = cls.__get_generated_live_samples(path_to_live_data, k, need_shuffle)
        elif type is KindOfData.ALL:
            # generate samples of both type
            result1 = cls.__get_generated_audioset_samples(n_samples, k)
            result2 = cls.__get_generated_live_samples(path_to_live_data, k, need_shuffle)
            result = {}
            # mixed samples
            for key in result1:
                data1, mask1 = result1[key]
                data2, mask2, _ = result2[key]  # ignore genres as there are no genres in audioset subset
                data = np.vstack([data1, data2])
                mask = np.vstack([mask1, mask2])
                # shuffle embeddings and masks of different sets in the same order
                if need_shuffle:
                    data, mask = cls.__shuffle(data, mask)
                result[key] = (data, mask)
        #save if need
        if path_to_save:
            if not os.path.isdir(path_to_save):
                os.mkdir(path_to_save)
            with open(path_to_save + '/' + name + ".pkl", "wb") as file:
                pickle.dump(result, file)
        return result

    @classmethod
    def __get_generated_audioset_samples(cls, n_samples, k):
        """ Get embedding and mask of audioset data
            :param n_samples: the number of samples to generate;
            :param k: coefficients for the capacities of sets (train:val:test);
            :return: result: a dictionary containing train, validation and test audioset samples
        """

        # prepare data to generate samples if necessary
        if cls.__music_data is None:
            cls.__generate_audioset_sample(n_samples)

        # generate samples
        n_music = len(cls.__music_data)
        n_speech = len(cls.__speech_data)
        data_train, mask_train = cls.__generate_data_sample(n_music_min=0, n_music_max=int(n_music * k[0]),
                                                            n_speech_min=0, n_speech_max=int(k[0] * n_speech),
                                                            N_samples=int(n_samples * k[0]))
        data_val, mask_val = cls.__generate_data_sample(n_music_min=int(n_music * k[0]),
                                                        n_music_max=int(n_music * k[1]),
                                                        n_speech_min=int(k[0] * n_speech),
                                                        n_speech_max=int(k[1] * n_speech),
                                                        N_samples=int(n_samples * (k[1] - k[0])))
        data_test, mask_test = cls.__generate_data_sample(n_music_min=int(n_music * k[1]), n_music_max=n_music,
                                                          n_speech_min=int(k[1] * n_speech), n_speech_max=n_speech,
                                                          N_samples=int(n_samples * (k[2] - k[1]) + 1))
        result = {'train': (data_train,mask_train),
                    'val': (data_val, mask_val),
                    'test': (data_test, mask_test)}
        return result

    @classmethod
    def __get_generated_live_samples(cls, path, k, need_shuffle):
        """ Get embedding and mask of live data (Youtube video, film etc)
                   :param path: pkl file containing information about masks and embeddings of samples from live set;
                   :param k: coefficients for the capacities of sets (train:val:test);
                   :param need_shuffle: flag, True if needed shuffling;
                   :return: result: a dictionary containing train, validation and test live samples
               """

        # prepare data to generate samples if necessary
        if cls.__mask_live is None:
            cls.__generate_live_sample(path, need_shuffle=need_shuffle)
        # generate samples
        n = len(cls.__data_live)
        data_train, mask_train, genres_train = cls.__data_live[0: int(k[0] * n)], \
                                               cls.__mask_live[0: int(k[0] * n)], cls.__genre_list[0: int(k[0] * n)]
        data_val, mask_val, genres_val = cls.__data_live[int(k[0] * n): int(k[1] * n)], \
                                         cls.__mask_live[int(k[0] * n): int(k[1] * n)], \
                                         cls.__genre_list[int(k[0] * n): int(k[1] * n)]
        data_test, mask_test, genres_test = cls.__data_live[int(k[0] * n): int(k[1] * n)],\
                                            cls.__mask_live[int(k[0] * n): int(k[1] * n)],\
                                            cls.__genre_list[int(k[0] * n): int(k[1] * n)]
        result = {'train': (data_train, mask_train, genres_train),
                  'val': (data_val, mask_val, genres_val),
                  'test': (data_test, mask_test, genres_test)}
        return result

    @classmethod
    def __generate_audioset_sample(cls, n_samples):
        """Generate n samples using AudioSet
        :param n_samples: the number of samples to generate;
        :return: void
        """
        # extract info from audio set
        assert os.path.isfile(cls.__path_to_audioset_data), "File {} not found".format(cls.__path_to_audioset_data)
        assert os.path.isfile(cls.__path_to_labels), "File {} not found".format(cls.__path_to_labels)
        cls.__music_data, cls.__speech_data = cls.__extract_info_audioset(cls.__path_to_audioset_data, cls.__path_to_labels)

        # extract info about music with genres classification
        assert os.path.isfile(cls.__path_to_genres), "File {} not found".format(cls.__path_to_genres)
        music_data_new_res = cls.__extract_music_for_genres(cls.__path_to_genres)
        # add data to sample
        cls.__music_data = np.vstack([cls.__music_data, music_data_new_res])

    @classmethod
    def __generate_live_sample(cls, path, need_shuffle):
        """Generate samples from YouTube video, films etc
                :param path: pkl file containing information about masks and embeddings of samples from live set
                :need_shuffle: flag, True if shuffling data is needed
                :return: void
        """
        # read data
        assert os.path.isfile(path), "File {} not found".format(path)
        with open(path, "rb") as handle:
            data_dict = pickle.load(handle)
        live_embed = cls.__uint8_to_float32([data_dict['embeddings_list']][0])
        mask_list = cls.__bool_to_float32([data_dict['mask_list']][0])
        genre_list = cls.__uint8_to_float32([data_dict['genre_mask_list']][0])

        #? why?
        if live_embed.shape[1] != 100:
            live_embed = cls.__x_reshape(live_embed)

        mask_list = cls.__mask_scaling(mask_list, 100)
        genre_list = cls.__mask_scaling(genre_list, 100)
        # shuffle embeddings and masks in the same order
        if need_shuffle:
            cls.__data_live, cls.__mask_live, cls.__genre_list = cls.__shuffle(live_embed, mask_list, genre_list)
        else:
            cls.__data_live, cls.__mask_live, cls.__genre_list = (live_embed, mask_list, genre_list)
        # np.zeros((N_samples, seq_len, 1), dtype=np.float32)
        sh = cls.__mask_live.shape
        cls.__mask_live = cls.__mask_live.reshape(sh[0], sh[1], 1)
        cls.__genre_list = cls.__genre_list.reshape(sh[0], sh[1], 1)

    @staticmethod
    def __x_reshape(live_embed):
        new_embed = np.zeros([live_embed.shape[0], 100, live_embed.shape[2]], dtype=float)
        for i in range(len(live_embed)):
            new_embed[i] = np.append(live_embed[i], [live_embed[i][-1, :]], axis=0)
        return new_embed

    @staticmethod
    def __mask_scaling(mask, new_size):
        new_mask = np.zeros([mask.shape[0], new_size], dtype=np.float32)
        ratio = new_size/mask.shape[1]

        for i, slice in enumerate(mask):
            index1 = np.argwhere(slice != 0)
            if len(index1) == 0:
                continue
            mask_class = slice[index1[0, 0]]
            index2 = 0
            new_slice = np.zeros(new_size, dtype=np.float32)
            while len(np.argwhere(slice[index2:] == mask_class)) != 0:
                index1 = index2 + np.argwhere(slice[index2:] == mask_class)[0][0]
                if len(np.argwhere(slice[index1:] == 0)) == 0:
                    new_slice[int(index1 * ratio):] = mask_class
                    break
                index2 = index1 + np.argwhere(slice[index1:] == 0)[0][0]
                new_slice[int(index1 * ratio):int(index2 * ratio)] = mask_class
            new_mask[i] = new_slice
        return new_mask



    @staticmethod
    def __shuffle(arr1, arr2, arr3=None):
        """Shuffles the arrays in the same order
            :param arr1: numpy array to shuffling
            :param arr2: numpy array to shuffling
            :param arr3: numpy array to shuffling
            :return (new_arr1, new_arr2, new_arr3): arrays shuffled in the same order
        """
        order = np.random.permutation(arr1.shape[0])
        new_arr1 = np.zeros(arr1.shape)
        new_arr2 = np.zeros(arr2.shape)
        new_arr3 = np.zeros(arr2.shape)
        for i in range(len(order)):
            new_arr1[order[i]] = arr1[i]
            new_arr2[order[i]] = arr2[i]
            if arr3 is not None:
                new_arr3[order[i]] = arr3[i]
        if arr3 is not None:
            return new_arr1, new_arr2, arr3
        else:
            return new_arr1, new_arr2

    @classmethod
    def __generate_data_sample(cls, n_music_min,n_music_max, n_speech_min, n_speech_max,
                             N_samples=10000, seq_len=100):
        """Generate random samples by mixing music and speech
            :param n_music_min: index of the first sound piece to be used (in music class)
            :param n_music_max: index of the last sound piece to be used (in music class)
            :param n_speech_min: index of the first sound piece to be used (in speech class)
            :param n_speech_max: index of the last sound piece to be used (in speech class)
            :param n_samples: the number of samples to generate
            :param seq_len: sequence length in seconds, the max length of music part in a single sample
            :return (data_sample, mask_sample) where
                data_sample: numpy tensor of samples with speech and music parts, shape is (sample_index, time, embeddings)
                mask_sample: numpy tensor with masks of the samples with ones in place of music part
        """
        number_tracks = int(round(seq_len) / 10)
        data_sample = np.zeros((N_samples, seq_len, param.EMBEDDING_SIZE), dtype=np.float32)
        mask_sample = np.zeros((N_samples, seq_len, 1), dtype=np.float32)
        for i in tqdm(range(N_samples)):
            # generat audio features
            for k in range(number_tracks):
                rand_idx = np.random.randint(n_speech_min, n_speech_max)
                data_sample[i, 10 * k:10 * (k + 1), :] = cls.__speech_data[rand_idx, :, :]
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
                    rand_idx = np.random.randint(n_music_min, n_music_max)
                    data_sample[i, start:start + 10, :] = cls.__music_data[rand_idx, :, :]
                    start += 10
                    curr_music_len = finish - start
                rand_idx = np.random.randint(n_music_min, n_music_max)
                data_sample[i, start:finish, :] = cls.__music_data[rand_idx, :curr_music_len, :]
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
        embeddings_list = data_dict_music["embedings_list"]

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
    # DataGenerator.get_generated_sample(KindOfData.AUDIOSET, [5,3,1], 1000)
    # args of cmd
    parser = argparse.ArgumentParser(description='Preprocessing')

    parser.add_argument('-t', type=str, action="store", dest="type", help='type of set to det. as - audioset, ls - live set, all - both')
    parser.add_argument('-ptdl', type=str, action="store", dest="ptdl", help='path to data about live sample')
    parser.add_argument('-pts', type=str, action="store", dest="pts", help='path to folder to save test, val and train samples')
    parser.add_argument('-r', type=str, action="store", dest="relation", help='relation for the capacities of sets (test:vel:train)')
    parser.add_argument('-n', type=int, action="store", dest="n_samples", help='amount of samples to generate')

    args = parser.parse_args()
    #generat samples depending on the type parameter
    if args.type == "as":
        type = KindOfData.AUDIOSET
    elif args.type == "ls":
        type = KindOfData.LIVE
    elif args.type == "all":
        type = KindOfData.ALL
    else:
        print("Unknown type: -t ")
    #parse relation
    r = list(map(int, args.relation.split(',')))
    s = sum(r)
    k = [r[0] / s]
    k.append(k[0] + r[1] / s)
    k.append(k[1] + r[2] / s)
    #generate data
    DataGenerator.get_generated_sample(type, r, args.n_samples, args.pts)
