# song-cutter
Pipeline for media content segmentation

Modules:
##Preprocessing
##FeatureExtractor
input: .pkl with files_list  
output: input.pkl + embeddings_list  
####func:
    get_audioset_features(path_pkl, path_to_save_pkl)  
* input: path_pkl - path to .pkl file with a list of files  
* output: path_to_save_pkl - path to save the result (input.pkl + embeddings)  
####guide for replacement model
The model is a class with one mandatory method:

    get_embeddings_list(self, files_list)
* input: files_list - list of files to process
* output: return value - list of embeddings corresponding to input files  

To select the desired model you just need to select the desired class. For example:

    model = VGGishModel.VGGishModel()
##DataGenerator
##Segmentation