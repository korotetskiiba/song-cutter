# song-cutter
Pipeline for media content segmentation

Modules:
## Preprocessing
## FeatureExtractor
## DataGenerator
input: type of generated set,
path to pkl file with data about live sample, 
path to folder to save test, val and train samples,
relation for the capacities of sets (test:vel:train),
amount of samples to generate

output: dictionary with keys('test', 'train', 'val') and tuple values (embedding, mask)  
#### func:
    get_generated_sample(cls, type: KindOfData, relation: list, n_samples:int=10000, 
                                path_to_live_data=None, path_to_save=None, name='sets')
    
* input: type- type of set to get, 
for the capacities of sets (train:val:test),
the number of samples to generate,
peth to pkl file containing information about masks and embeddings of samples from live set,
path to folder for saving samples, name of file to save samples
* output: a dictionary containing train, validation and test samples


#### Example
As an example, if you want to generate 10,000 samples from an audio set and get samples in the ratio 5:3:1 and save data in 'output_folder', you should call the following function:


```
result = DataGenerator.get_generated_sample(KindOfData.AUDIOSET, [5,3,1], 10000,path_to_save='outout_folder')
```

#### CLI
It's possible to run the module using the command prompt as well. In that case, the program parameters should be as follows:
  * t TYPE       type of set to det. as - audioset, ls - live set, all - both
  * ptdl PTDL    path to data about live sample
  * pts PTS      path to folder to save test, val and train samples
  * r RELATION   relation for the capacities of sets (test:vel:train)
  * n N_SAMPLES  amount of samples to generate

#### CLI example
So, to do the same action via the command prompt and save sets in folder 'sets' you should:
```
python DataGenerator.py -t as -n 10000 -r 5,3,1 -pts outout_folder
```

## Segmentation: