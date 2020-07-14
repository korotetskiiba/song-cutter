# song-cutter
Pipeline for media content segmentation

Modules:
 - Preprocessing
 - FeatureExtractor
 - DataGenerator
 - Segmentation
## Preprocessing
* input: 
	- meta-info txt-file
	- video mp4-file (video can also be downloaded from YT if given a link in the 1st line of meta-info file).
* output: 
	- for train-pipeline - .plk file with files-list(path to samples obtained from given video) and mask(binary mask of the whole audio)
	- for inference-pipeline - wav-audiotrack of the given video.
#### funcs:
```
convert_to_wav(path_to_video, path_to_audio)
```
Converts given video to .wav-audio.
* input:
	 - `path_to_video` - path to the video to be converted to audio
	 - `path_to_audio` - path to where audio is to be placed after conversion
* output:
	- wav-audio at path_to_audio.
```
preprocess_train(path_to_meta, path_to_video, path_to_audio, seq_len=100)
```
This method is to be used when training the model.
Takes the video either from the given location or downloads using the link, converts it to audio,
cuts audio in pieces, translates the segments-info into a binary mask. Creates a pkl file with paths
to all audio-cuts and the binary-mask.
* input: 
	 - `path_to_meta` - path to meta-info about the audio(video)
	 - `path_to_video` - path to the existing video to be converted to training data (if it doesn't exist, it can be downloaded from YouTube using the link given in 1st line of meta-info file)
	 - `path_to_audio` - path to where audio is to be placed after conversion
	 - `seq_len` - the length of sample in seconds to which the audio is cut
* output:
	- `pickle-file` - .pkl file with dict{"files_list": <list of paths to cuts>, "mask_list": <bin mask for the whole audio>}. .pkl file is placed in the same dir as the initial audio and named the same way

#### CLI
This module can be called through CLI, see options at -h(--help).
Optional arguments:
  * -i MODE
  	- mode=train if module is part of train-pipeline. mode=predict if module is part of inference-pipeline.
  * -v PATH_TO_VIDEO  
  	- absolute path to the video-file including its name and format. if the link is given in meta-info file, this would be the path to save the video(if its not yet downloaded to this location).
  * -a PATH_TO_AUDIO  
  	- absolute path to the audio-file including its name and format. this is the where the audio file is saved after being converted.
  * -m PATH_TO_META   
  	- absolute path to the meta-file including its name and format.
  * -s SEQ_LEN        
  	- the length of the training sample in seconds. given video's audiotrack is cut in pieces of this size. default=100
  * -h, --help        
  	- show this help message

#### CLI example:

To run as part of train-pipeline:
```
python PreprocessingModule.py -i train -v path_to_video -a path_to_audio -m path_to_meta
```
To run as part of inference-pipeline:
```
python PreprocessingModule.py -i predict -v path_to_video -a path_to_audio
```
 
## FeatureExtractor
 * input: 
 	- .pkl with files_list  
 * output: 
	- input.pkl + embeddings_list  
#### func:  
```
get_audioset_features(path_pkl, path_to_save_pkl)
```
* input:
	 - `path_pkl` - path to .pkl file with a list of files  
* output:
	 - `path_to_save_pkl` - path to save the result (input.pkl + embeddings)  

#### guide for replacement model:
The model is a class with one mandatory method:  
```
get_embeddings_list(self, files_list)
```
* input:
	- `files_list` - list of files to process
* output:
	 - return value - list of embeddings corresponding to input files  

To select the desired model you just need to select the desired class. For example:  
`model = VGGishModel.VGGishModel()`

## DataGenerator
 - input: 
 	- type of generated set
	- path to pkl file with data about live sample
	- path to folder to save test, val and train samples
	- relation for the capacities of sets (test:vel:train)
	- amount of samples to generate

 - output: 
 	- dictionary with keys('test', 'train', 'val') and tuple values (embedding, mask)  
#### func:
```
get_generated_sample(cls, type: KindOfData, relation: list, n_samples:int=10000, 
                                path_to_live_data=None, path_to_save=None, name='sets')
```
    
* input: 
	- `type`- type of set to get 
	- `relation` - for the capacities of sets (train:val:test)
	- `n_samples` - the number of samples to generate
	- `path_to_live_data` - path to pkl file containing information about masks and embeddings of samples from live set
	- `path_to_save` - path to folder for saving samples
	- `name` - name of file to save samples
* output:
	- a dictionary containing train, validation and test samples


#### Example:
As an example, if you want to generate 10,000 samples from an audio set and get samples in the ratio 5:3:1 and save data in 'output_folder', you should call the following function:


```
result = DataGenerator.get_generated_sample(KindOfData.AUDIOSET, [5,3,1], 10000,path_to_save='outout_folder')
```

#### CLI
It's possible to run the module using the command prompt as well. In that case, the program parameters should be as follows:
  * t TYPE       
  	- type of set to det. as - audioset, ls - live set, all - both
  * ptdl PTDL    
  	- path to data about live sample
  * pts PTS      
  	- path to folder to save test, val and train samples
  * r RELATION   
  	- relation for the capacities of sets (test:vel:train)
  * n N_SAMPLES  
  	- amount of samples to generate

#### CLI example:
So, to do the same action via the command prompt and save sets in folder 'sets' you should:
```
python DataGenerator.py -t as -n 10000 -r 5,3,1 -pts outout_folder
```

## Segmentation
 - input: data tensors (`x_train`, `x_valid`, `x_test`, `y_train`, etc.) of shape `(num_samples, time, embeddings)`, model checkpoint (.h5 file)
 - output: model checkpoint (.h5 file), list of time intervals (interval is the list of the beginning and the end time in format 'hh:mm:ss'), evaluation summary (plots and metrics) and video (or audio) slices according to the prediction intervals
### funcs:
implemented methods:
- `__init__(self)`: initialize `SegmentationModule` block
- `exec_fit(self, x_train, x_valid, y_train, y_valid, checkpoint_file, epochs=30, batch_size=32)`: builds default model if it hasn't been built yet and executes model fit on passed data tensors
	input: 
	- `x_train` - train data of shape `(num_samples, time, embeddings)`
	- `x_valid` - validation data of the same shape
	- `y_train` - train data labels of shape `(num_samples, time, 1)`
	- `y_valid` - validation data of the same shape
	- `checkpoint_file` - file to save checkpoints (has '.h5' extension)
	- `epochs` - the number of epochs to fit
	- `batch_size` - the batch size to fit
- `load_from_checkpoint(self, checkpoint_file)`: load model from checkpoint (consider it is default CRF model)
	input:
	- `checkpoint_file` - file to load model weights (has '.h5' extension)
- `predict(self, x_data)`: predict label for passed data tensor
	input:
	- `x_data` - data tensor of shape `(num_samples, time, embeddings)`
	output: the list of interval in the format mentioned above
- `get_model(self)`: get current keras model used in the module to predict and evaluate
	output: keras model
- `cut_file(path_to_file, target_path, prediction_intervals)`: cut video or sound file to slices mentioned in `prediction_intervals`.
	input:
	- `path_to_file` - path to video or sound file (extension must be '.mp4' or '.wav')
	- `target_path` - path to target directory to save slices, with name prefix used for each slice
	- `prediction_intervals` - the list of time intervals got from `predict(self, x_data)`
- `evaluate(self, x_test, y_test, target_path)`: evaluate model (count metrics, draw ROC curve plot, draw plot with the ground truth mask and predicted mask)
	input:
	- `x_test` - data tensor of shape `(1, time, embeddings)`
	- `y_test` - the ground truth tensor of shape `(time, )`
	- `target_path` - directory where plots and metrics will be saved

### Usage example:

import module:

```
from SegmentationModule import SegmentationModule as SMod
```

init and fit for 20 epochs, save model to `test_checkpoint.h5` file:

```
segm_module = SMod()

segm_module.exec_fit(x_train, x_valid, y_train, y_valid, 
                    "test_checkpoint.h5", epochs=20)
```

load pre-trained model from checkpoint:

```
segm_module.load_from_checkpoint("test_checkpoint.h5")
```

make prediction and slice video according to the prediction:

```
intervals = segm_module.predict(test_sample)

segm_module.cut_file(video/videofile.mp4", "video/cut", intervals)
```

evaluate: calculate and save metrics, draw plots:

```
save_file = "evaluate_1"

segm_module.evaluate(x_sample, ground_truth, save_file)
```