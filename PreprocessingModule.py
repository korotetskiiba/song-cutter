from moviepy.editor import *
from pytube import YouTube
from datetime import datetime
import pickle


class PreprocessingModule:

    @staticmethod
    def convert_to_wav(path_to_video, path_to_audio):
        """
        Converts given video to .wav-audio.

        :param path_to_video: path to the video to be converted to audio;
        :param path_to_audio: path to where audio is to be placed after conversion;
        :return: void
        """
        with VideoFileClip(path_to_video) as videoclip:
            audioclip = videoclip.audio
            audioclip.write_audiofile(path_to_audio, logger=None)

    @staticmethod
    def __cut_in_pieces(path_to_audio, seq_len):
        """
        Cuts audio in disjoint pieces of fixed size starting from the very beginning.
        The last piece is not included in the result if its length != seq_len.
        The cuts are placed in the same dir as the initial audio-file.

        :param path_to_audio: path to the audio to be cut in pieces;
        :param seq_len: length of a desired cut;
        :return: list of all paths to cuts.
        """
        assert os.path.isfile(path_to_audio), "Audio file {} not found".format(path_to_audio)

        audiofile_name = os.path.basename(path_to_audio).split(".")[0]
        path_to_cuts = os.path.dirname(path_to_audio)

        paths_to_cuts = []
        with AudioFileClip(path_to_audio) as audio:
            num_cuts = int(audio.duration//seq_len)
            for cut in range(1, num_cuts + 1):
                path_to_cut = path_to_cuts + "/" + audiofile_name + str(cut) + ".wav"
                audio.subclip((cut-1) * seq_len, cut * seq_len).write_audiofile(
                    path_to_cut,
                    logger=None
                )
                paths_to_cuts.append(path_to_cut)
        return paths_to_cuts

    @staticmethod
    def __generate_pkl(paths_to_cuts, bin_mask):
        """
        Generates pickle-file - .pkl file with
        dict{"files_list": <list of paths to cuts>, "mask": <bin mask for the whole audio>}.
        .pkl file is placed in the same dir as the initial audio and named the same way.

        :param paths_to_cuts: list with all paths to cuts, that were generated from audio-file;
        :param bin_mask: binary mask of the whole audio-file;
        :return: void.
        """
        pkl_dict = {"files_list": paths_to_cuts, "mask": bin_mask}
        pkl_name = os.path.abspath(paths_to_cuts[0]).replace("1.wav", "") + ".pkl"
        with open(pkl_name, "wb") as pkl:
            pickle.dump(pkl_dict, pkl)

    @staticmethod
    def __download_from_youtube(link, path_to_video):
        """
        Downloads video from YouTube using the link (found beforehand in the 1st line of meta-info file).

        :param link: YouTube link to the video;
        :param path_to_video: path to where video is to be placed after download;
        :return: void
        """
        yt = YouTube(link)
        yt.streams.first().download(path_to_video)

    @staticmethod
    def __preprocess_meta(path_to_meta):
        """
        Converts meta-info into a unified binary format,
        where for each second "1" - "song", "0" - "not song".
        Extracts YouTube link if given in the 1st line of the meta-info file.

        :param path_to_meta: path to meta-info about the audio;
        :return: binary mask generated based on given meta.
        """
        assert os.path.isfile(path_to_meta), "Meta-info file {} not found".format(path_to_meta)

        bin_mask = []
        with open(path_to_meta) as meta:
            for line in meta:
                link = line.rstrip("\n") if line.startswith("http") else None
                if link:
                    continue
                start, end, label = line.split(",")
                start = datetime.strptime(start, '%H:%M:%S')
                end = datetime.strptime(end, '%H:%M:%S')
                start_sec = PreprocessingModule.__to_secs(start)
                end_sec = PreprocessingModule.__to_secs(end)
                duration = (end_sec - start_sec)
                for _ in range(duration):
                    bin_mask.append(1 if label == "song\n" else 0)
        assert len(bin_mask) == end_sec, "Mask is not correct!"
        return link, bin_mask

    @staticmethod
    def __to_secs(time):
        total_seconds = time.second + time.minute * 60 + time.hour * 3600
        return total_seconds

    @staticmethod
    def preprocess_train(path_to_meta, path_to_video, path_to_audio, seq_len=100):
        """
        This method is to be used when training the model.
        :param path_to_meta: path to meta-info about the audio(video);
        :param path_to_video: path to the existing video to be converted to training data
                              (if it doesn't exist can be downloaded from YouTube using the
                              link given in 1st line of meta-info file);
        :param path_to_audio: path to where audio is to be placed after conversion;
        :param seq_len: the length of sample in seconds to which the audio is cut;
        :return: void.
        """
        assert os.path.isfile(path_to_meta), "Meta-info file {} not found".format(path_to_meta)
        assert os.path.isfile(path_to_audio), "Audio file {} not found".format(path_to_audio)

        link, bin_mask = PreprocessingModule.__preprocess_meta(path_to_meta)
        if link and not os.path.isfile(path_to_video):
            PreprocessingModule.__download_from_youtube(link, path_to_video)
        PreprocessingModule.convert_to_wav(path_to_video, path_to_audio)
        files_list = PreprocessingModule.__cut_in_pieces(path_to_audio, seq_len)
        PreprocessingModule.__generate_pkl(files_list, bin_mask)
        
