import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os

initial_signal = {}
initial_ica_composition = {}
initial_ica_signal = {}
initial_sample_rate = {}
initial_sens_lab = {}
initial_signal_time = {}
initial_signal_markers = {}
initial_total_epochs = {}
initial_deleted_epochs = {}


component_names = ["Component " + str(n) for n in range(17)]
component_names.extend(["ECG", "EOG-R", "EOG-L", "EMG"])


if __name__ == "__main__":
    for stimulation in [StimualtionMarker.ANODAL, StimualtionMarker.CATHODAL, StimualtionMarker.SHAM]:
        [initial_signal[stimulation],
         initial_ica_composition[stimulation],
         initial_ica_signal[stimulation],
         initial_sample_rate[stimulation],
         initial_sens_lab[stimulation],
         initial_signal_time[stimulation],
         initial_signal_markers[stimulation],
         initial_deleted_epochs[stimulation],
         initial_total_epochs[stimulation]] = load_and_process_data(stimulation)

    tf.reset_default_graph()
    embedding_var = tf.Variable(X_cut, dtype=tf.float32, name='embedding')

    with open('/Users/alexanderashikhmin/tsne data/meta.tsv', 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(y_cut):
            f.write("%d\t%s\n" % (index, label))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        LOG_DIR = '/Users/alexanderashikhmin/tsne data'

        #     saver = tf.train.Saver()
        #     saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), 1)

        # Use the same LOG_DIR where you stored your checkpoint.
        summary_writer = tf.summary.FileWriter(LOG_DIR, graph=session.graph)

        # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
        config = projector.ProjectorConfig()

        # You can add multiple embeddings. Here we add only one.
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        #     Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = os.path.join(LOG_DIR, 'meta.tsv')

        # Saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(summary_writer, config)
        summary_writer.flush()

        saver = tf.train.Saver()
        saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), 1)