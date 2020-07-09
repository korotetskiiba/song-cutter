import subprocess
from typing import List


class PreprocessingModule:
    def __init__(self, converter):
        """
        Constructor of the preprocessing module.
        
        :param converter: path to the executable ffmpeg converter.
        """
        self.converter = converter

    def convert_to_wav(self, path_to_video, path_to_audio):
        """
        Converts given video to .wav-audio.

        :param path_to_video: path to the video to be converted to audio;
        :param path_to_audio: path to where audio is to be placed after conversion;
        :return: report-info given by the converter, this is to be explored in case of unexpected behaviour.
        """
        command = "{} -i {} -ab 160k -ac 2 -ar 44100 -vn {}".format(self.converter, path_to_video, path_to_audio)
        res = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res

    def cut_in_pieces(self, path_to_audio, path_to_cuts, path_to_meta, seq_len=100):
        """
        Cuts audio in disjoint pieces of fixed size starting from the very beginning.
        The last piece is not included in the result if its length != seq_len.
        Given meta-info is transformed into .pkl file with
        dict{files_list: <list of paths to cuts>, mask: <bin mask for the whole audio>}.
        .pkl file with meta is placed in the same dir as initial meta and named the same way.

        :param path_to_audio: path to the audio to be cut in pieces;
        :param path_to_cuts: path to where cuts are to be placed;
        :param path_to_meta: path to meta-info about the audio;
        :param seq_len: length of a desired cut;
        :return: report-info given by the converter, this is to be explored in case of unexpected behaviour.
        """
        pass

    @classmethod  # @staticmethod (?)
    def __preprocess_meta(path_to_meta, subs=False) -> List:
        """
        Converts different meta-info types (subtitles and segment) into a unified binary format,
        where for each second "1" - "song", "0" - "not song".

        :param path_to_meta: path to meta-info about the audio;
        :param subs: reflects if meta is a subtitle-file, by default assume meta is given in .segments format;
        :return: binary mask generated based on given meta.
        """
        pass



