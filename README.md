# song-cutter
Pipeline for media content segmentation

Modules:
 - Preprocessing
 - FeatureExtractor
 - DataGenerator
 - Segmentation
## Preprocessing
* input: meta-info txt-file, video mp4-file (video can also be downloaded from YT if given a link in the 1st line of meta-info file).
* output: for train-pipeline - .plk file with files-list(path to samples obtained from given video) and mask(binary mask of the whole audio),
        for inference-pipeline - wav-audiotrack of the given video.
#### func:
    convert_to_wav(path_to_video, path_to_audio)
Converts given video to .wav-audio.
* input: path_to_video - path to the video to be converted to audio,
         path_to_audio - path to where audio is to be placed after conversion.
* output: wav-audio at path_to_audio.
#### func:
    preprocess_train(path_to_meta, path_to_video, path_to_audio, seq_len=100)
This method is to be used when training the model.
Takes the video either from the given location or downloads using the link, converts it to audio,
cuts audio in pieces, translates the segments-info into a binary mask. Creates a pkl file with paths
to all audio-cuts and the binary-mask.
* input: path_to_meta - path to meta-info about the audio(video),
         path_to_video - path to the existing video to be converted to training data (if it doesn't exist, it can be downloaded from YouTube using the link given in 1st line of meta-info file),
         path_to_audio - path to where audio is to be placed after conversion,
         seq_len - the length of sample in seconds to which the audio is cut.
* output: pickle-file - .pkl file with dict{"files_list": <list of paths to cuts>, "mask": <bin mask for the whole audio>}. .pkl file is placed in the same dir as the initial audio and named the same way.
 
## FeatureExtractor
## DataGenerator
## Segmentation