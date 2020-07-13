# song-cutter
Pipeline for media content segmentation

Modules:
## Preprocessing
## FeatureExtractor
## DataGenerator
## Segmentation
input: data tensors (`x_train`, `x_valid`, `x_test`, `y_train`, etc.) of shape `(num_samples, time, embeddings)`, model checkpoint (.h5 file)
output: model checkpoint (.h5 file), list of time intervals (interval is the list of the beginning and the end time in format 'hh:mm:ss'), evaluation summary (plots and metrics) and video (or audio) slices according to the prediction intervals
### funcs
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