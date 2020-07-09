import tensorflow as tf
import pickle
from vggish_files import vggish_postprocess, vggish_slim, vggish_input, vggish_params



class Feature_extraction:

    @staticmethod
    def get_audioset_features(path_pkl, path_to_save_pkl):
        with open(path_pkl, "rb") as handle:
            data_dict_music = pickle.load(handle)

        file_list = data_dict_music["files_list"]

        # Prepare a postprocessor to munge the model embeddings.

        pproc = vggish_postprocess.Postprocessor(vggish_params.PCA_DUMP)

        with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
            # Define the model in inference mode, load the checkpoint, and
            # locate input and output tensors.
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(sess, vggish_params.VGGISH_CPT)
            features_tensor = sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
            embedding_tensor = sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)

            # Run inference and postprocessing.
            embed_list = []
            for f in file_list:
                examples_batch = vggish_input.wavfile_to_examples(f)
                [embedding_batch] = sess.run([embedding_tensor],
                                             feed_dict={features_tensor: examples_batch})
                postprocessed_batch = pproc.postprocess(embedding_batch)
                embed_list.append(postprocessed_batch)
                #print(f)


            data_dict_music["embedings_list"] = embed_list
            with open(path_to_save_pkl, "wb") as handle:
                pickle.dump(data_dict_music, handle)
