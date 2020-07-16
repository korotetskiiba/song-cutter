from moviepy.editor import VideoFileClip, AudioFileClip
from pytube import YouTube
from datetime import datetime
import os
import argparse
import pickle


class PreprocessingModule:

    @staticmethod
    def __convert_one_to_wav(path_to_video, path_to_audio):
        """
        Converts given video to .wav-audio.
        
        :param path_to_video: path to the video to be converted to audio;
        :param path_to_audio: path to where audio is to be placed after conversion;
        :return: void
        """
        if os.path.isfile(path_to_audio):
            return
        with VideoFileClip(path_to_video) as videoclip:
            audioclip = videoclip.audio
            audioclip.write_audiofile(path_to_audio, logger=None)

    @staticmethod
    def __convert_dir_to_wav(path_to_video_dir, path_to_audio_dir):
        """
        Converts all videos in a given folder to .wav-audio.
        
        :param path_to_video_dir: dir with videos to be converted;
        :param path_to_audio_dir: path to dir where audios are to be placed after conversion;
        :return: void
        """
        if not os.path.isdir(path_to_audio_dir):
            os.mkdir(path_to_audio_dir)
        video_paths = PreprocessingModule.__find_format(path_to_video_dir, ".mp4")
        for path_to_video in video_paths:
            name = str(os.path.basename(path_to_video).split(".")[0])
            path_to_audio = os.path.join(path_to_audio_dir, name + ".wav")
            PreprocessingModule.__convert_one_to_wav(path_to_video, path_to_audio)

    @staticmethod
    def convert_to_wav(path_to_video, path_to_audio):
        """
        Converts given video or dir with videos to .wav-audio(s).
        
        :param path_to_video: path to the video(s) to be converted to audio(s) (or path to dir with videos);
        :param path_to_audio: path to where audio(s) is(are) to be placed after conversion (or path to dir) ;
        :return: void
        """
        if os.path.isdir(path_to_video) and os.path.isdir(path_to_audio):
            PreprocessingModule.__convert_dir_to_wav(path_to_video, path_to_audio)
        if os.path.isfile(path_to_video):
            PreprocessingModule.__convert_one_to_wav(path_to_video, path_to_audio)

    @staticmethod
    def __cut_in_pieces(path_to_audio, bin_mask, seq_len):
        """
        Cuts audio in disjoint pieces of fixed size starting from the very beginning.
        The last piece is not included in the result if its length != seq_len.
        The cuts are placed in the same dir as the initial audio-file.
        
        :param path_to_audio: path to the audio to be cut in pieces;
        :param bin_mask: binary mask of the whole audio-file;
        :param seq_len: length of a desired cut;
        :return: list of all paths to cuts, array of masks corresponding to those cuts.
        """
        assert os.path.isfile(path_to_audio), "Audio file not found"

        audiofile_name = str(os.path.basename(path_to_audio).split(".")[0])
        path_to_cuts = os.path.dirname(path_to_audio)

        paths_to_cuts = []
        mask_list = []
        with AudioFileClip(path_to_audio) as audio:
            num_cuts = int(audio.duration // seq_len)
            for cut in range(1, num_cuts + 1):
                path_to_cut = os.path.join(path_to_cuts, audiofile_name + str(cut) + ".wav")
                audio.subclip((cut - 1) * seq_len, cut * seq_len).write_audiofile(
                    path_to_cut,
                    logger=None
                )
                paths_to_cuts.append(path_to_cut)

                mask = bin_mask[(cut - 1) * seq_len: cut * seq_len]
                mask_list.append(mask)
        return paths_to_cuts, mask_list

    @staticmethod
    def __generate_pkl(paths_to_cuts, mask_list):
        """
        Generates pickle-file - .pkl file with
        dict{"files_list": <list of paths to cuts>, "mask_list": <bin masks for the cuts of the audio>}.
        .pkl file is placed in the same dir as the initial audio and named the "pickle_samples".
        
        :param paths_to_cuts: list with all paths to cuts, that were generated from audio-file;
        :param bin_mask: binary mask of the whole audio-file;
        :return: void.
        """
        pkl_dict = {"files_list": paths_to_cuts, "mask_list": mask_list}
        pkl_path = os.path.join(os.path.dirname(paths_to_cuts[0]), "pickle_samples.pkl")
        with open(pkl_path, "wb") as pkl:
            pickle.dump(pkl_dict, pkl)

    @staticmethod
    def __download_from_youtube(link, path_to_video):
        """
        Downloads video from YouTube using the link (found beforehand in the 1st line of meta-info file).
        
        :param link: YouTube link to the video;
        :param path_to_video: path to where video is to be placed after download;
        :return: void
        """
        dir = os.path.dirname(path_to_video)
        name = str(os.path.basename(path_to_video).split(".")[0])
        yt = YouTube(link)
        yt.streams.first().download(dir, filename=name)

    @staticmethod
    def __preprocess_meta(path_to_meta):
        """
        Converts meta-info into a unified binary format,
        where for each second "1" - "song", "0" - "not song".
        Extracts YouTube link if given in the 1st line of the meta-info file.
        
        :param path_to_meta: path to meta-info about the audio;
        :return: binary mask generated based on given meta and link to YouTube src if given.
        """
        assert os.path.isfile(path_to_meta), "Meta-info file {} not found".format(path_to_meta)

        bin_mask = []
        link = None
        with open(path_to_meta) as meta:
            for line in meta:
                if line.startswith("http"):
                    link = line.rstrip("\n")
                    continue
                start, end, label = line.split(",")
                start = datetime.strptime(start, '%H:%M:%S')
                end = datetime.strptime(end, '%H:%M:%S')
                start_sec = PreprocessingModule.__to_secs(start)
                end_sec = PreprocessingModule.__to_secs(end)
                duration = (end_sec - start_sec)
                for _ in range(duration):
                    bin_mask.append(1 if (label == "song\n" or label == "song") else 0)
        assert len(bin_mask) == end_sec, "Mask is not correct!"
        return link, bin_mask

    @staticmethod
    def __to_secs(time):
        total_seconds = time.second + time.minute * 60 + time.hour * 3600
        return total_seconds

    @staticmethod
    def __find_format(path, format):
        list_of_paths = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(format):
                    list_of_paths.append(os.path.join(root, file))
        return list_of_paths

    @staticmethod
    def __preprocess_one(path_to_meta, path_to_video, path_to_audio, seq_len=100):
        """
        Takes the video either from the given location or downloads using the link, converts it to audio,
        cuts audio in pieces, translates the segments-info into binary masks.
        
        :param path_to_meta: path to meta-info about the audio(video);
        :param path_to_video: path to the existing video to be converted to training data
                              (if it doesn't exist can be downloaded from YouTube using the
                              link given in 1st line of meta-info file);
        :param path_to_audio: path to where audio is to be placed after conversion;
        :param seq_len: the length of sample in seconds to which the audio is cut;
        :return: void.
        """
        assert os.path.isfile(path_to_meta), "Meta-info file not found"

        link, bin_mask = PreprocessingModule.__preprocess_meta(path_to_meta)
        if link and not os.path.isfile(path_to_video):
            PreprocessingModule.__download_from_youtube(link, path_to_video)
        PreprocessingModule.__convert_one_to_wav(path_to_video, path_to_audio)
        files_list, mask_list = PreprocessingModule.__cut_in_pieces(path_to_audio, bin_mask, seq_len)
        return files_list, mask_list

    @staticmethod
    def __preprocess_dir(path_to_meta_dir, path_to_video_dir, path_to_audio_dir, seq_len=100):
        """
        Takes the videos either from the given location or downloads using the link, converts them to audios,
        cuts audios in pieces, translates the segments-info into binary masks. Creates a pkl file with paths
        to all audio-cuts and the binary-masks of all cuts of all videos.
        .pkl file is placed in the same dir as the initial audios and named "pickle_samples".
        
        :param path_to_meta_dir: path to dir with meta-info files about the audios(videos);
        :param path_to_video_dir: path to dir with videos to be converted to training data
                              (if the video in the folder doesn't exist, it can be downloaded
                               from YouTube using the link given in 1st line of meta-info file);
        :param path_to_audio_dir: path to dir where audios are to be placed after conversion;
        :param seq_len: the length of sample in seconds to which the audios are cut;
        :return: void.
        """
        if not os.path.isdir(path_to_video_dir):
            os.mkdir(path_to_video_dir)
        if not os.path.isdir(path_to_audio_dir):
            os.mkdir(path_to_audio_dir)
        #  collect paths to all meta-info files in dir
        meta_paths = PreprocessingModule.__find_format(path_to_meta_dir, ".txt")
        files = []
        masks = []
        for path_to_meta in meta_paths:
            name = str(os.path.basename(path_to_meta).split(".")[0])
            path_to_video = os.path.join(path_to_video_dir, name + ".mp4")
            path_to_audio = os.path.join(path_to_audio_dir, name + ".wav")
            files_list, mask_list = PreprocessingModule.__preprocess_one(
                path_to_meta,
                path_to_video,
                path_to_audio,
                seq_len
            )
            files.extend(files_list)
            masks.extend(mask_list)
        PreprocessingModule.__generate_pkl(files, masks)


    @staticmethod
    def preprocess_train(path_to_meta, path_to_video, path_to_audio, seq_len=100):
        """
        This method is to be used when training the model.
        Takes the video(s) either from the given location or downloads using the link, converts it(them) to audio(s),
        cuts audio(s) in pieces, translates the segments-info into binary masks. Creates a pkl file with paths
        to all audio-cuts and the binary-masks.
        .pkl file is placed in the same dir as the initial audios and named "pickle_samples".
        
        :param path_to_meta: path to meta-info about the audio(video) (either a path to file or directory);
        :param path_to_video: path to video(s) to be converted to training data
                              (if video doesn't exist, it can be downloaded from YouTube using the
                              link given in 1st line of meta-info file) (either a path to file or directory);
        :param path_to_audio: path to where audio is to be placed after conversion (either a path to file or directory);
        :param seq_len: the length of sample in seconds to which the audio is cut;
        :return: void.
        """
        if os.path.isdir(path_to_meta):
            PreprocessingModule.__preprocess_dir(
                path_to_meta,
                path_to_video,
                path_to_audio,
                seq_len
            )
        else:
            files_list, mask_list = PreprocessingModule.__preprocess_one(
                path_to_meta,
                path_to_video,
                path_to_audio,
                seq_len
            )
            PreprocessingModule.__generate_pkl(files_list, mask_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing', add_help=False)

    parser.add_argument('-i', action="store", dest="mode", help="mode=train if module is part of train-pipeline."
                                                                "mode=predict if module is part of inference-pipeline.")
    parser.add_argument('-v', action="store", dest="path_to_video", help="absolute path to the video-file including "
                                                                         "its name and format (or abs.path to dir with "
                                                                         "videos). if the link is given in meta-info "
                                                                         "file, this would be the path to save the "
                                                                         "video (if its not yet downloaded to this "
                                                                         "location). if dir is given, each video "
                                                                         "is named the same as the corresponding meta.")
    parser.add_argument('-a', action="store", dest="path_to_audio", help="absolute path to the audio-file including its"
                                                                         " name and format (or abs.path to dir with "
                                                                         "audios). this is the where the "
                                                                         "audio file is saved after being converted. "
                                                                         "if dir is given, each audio "
                                                                         "is named the same as the corresponding meta.")
    parser.add_argument('-m', action="store", dest="path_to_meta", help="absolute path to the meta-file including "
                                                                        "its name and format (or abs.path to dir with "
                                                                        "meta-info files).")
    parser.add_argument('-s', action="store", dest="seq_len", help="the length of the training sample in seconds. "
                                                                   "given video's audiotrack is cut in pieces "
                                                                   "of this size. default=100")
    parser.add_argument('-h', '--help', action="help", help="show this help message")

    args = parser.parse_args()
    if args.seq_len is None:
        args.seq_len = 100

    if args.mode == "train":
        PreprocessingModule.preprocess_train(args.path_to_meta, args.path_to_video, args.path_to_audio, args.seq_len)
    if args.mode == "predict":
        PreprocessingModule.convert_to_wav(args.path_to_video, args.path_to_audio)
