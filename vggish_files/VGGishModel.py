from vggish_files import vggish_postprocess, vggish_slim, vggish_input, vggish_params
import tensorflow as tf


class VGGishModel:
    def __init__(self):
        # Prepare a postprocessor to munge the model embeddings.
        self.pproc = vggish_postprocess.Postprocessor(vggish_params.PCA_DUMP)

        with tf.Graph().as_default():
            # Define the model in inference mode, load the checkpoint, and
            # locate input and output tensors.
            self.sess = tf.compat.v1.Session()
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(self.sess, vggish_params.VGGISH_CPT)
            self.features_tensor = self.sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
            self.embedding_tensor = self.sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)


    def __del__(self):
        self.sess.close()

    def get_embeddings_list(self, files_list):
        # Run inference and postprocessing.
        embed_list = []
        for f in files_list:
            assert isinstance(f, str) and f.endswith('.wav'), "Wrong path to wav in files list"

            examples_batch = vggish_input.wavfile_to_examples(f)
            [embedding_batch] = self.sess.run([self.embedding_tensor], feed_dict={self.features_tensor: examples_batch})

            postprocessed_batch = self.pproc.postprocess(embedding_batch)
            embed_list.append(postprocessed_batch)

        return embed_list
