from data_input import count_records

import audio
import data_input
import os
import tensorflow as tf

SAVE_EVERY = 250

def check_tf_records_with_tensorboard(files, data_path, save_path):
    sr = 24000 if "vctk" in save_path else 16000
    meta = data_input.load_meta(data_path)
    ivocab = meta["vocab"]

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('log/' + save_path + '/debug', sess.graph)

        record_placeholder = tf.placeholder(tf.string)
        features_in = tf.parse_single_example(
            record_placeholder,
            features={
                "index": tf.FixedLenFeature([], tf.int64),
                "stfts": tf.FixedLenFeature((180, 2050), tf.float32),
                "stfts_shape": tf.FixedLenFeature((2), tf.int64),
                "mels": tf.FixedLenFeature((180, 160), tf.float32),
                "mels_shape": tf.FixedLenFeature((2), tf.int64),
                "texts": tf.VarLenFeature(tf.int64),
                "text_lens": tf.FixedLenFeature([], tf.int64),
                "speech_lens": tf.FixedLenFeature([], tf.int64),           
            })

        for file_path in files:
            print("Reading file: %s" % file_path)
            for record, i in zip(tf.python_io.tf_record_iterator(file_path), range(count_records([file_path]))):
                if i % SAVE_EVERY == 0 and (i != 0 or SAVE_EVERY == 1):
                    print("Iteration %d" % i)
                    features = sess.run(features_in, feed_dict={record_placeholder: record})
                    texts = features["texts"]
                    texts = tf.sparse_to_dense(texts.indices, texts.dense_shape, texts.values)

                    # Convert integers to words
                    texts = "".join(filter(lambda x: x != "<pad>", [ivocab[word] for word in texts.eval(session=sess)]))
                    print("Texts: %s" % texts)

                    print("stfts shape: %s" % str(features["stfts"].shape))

                    print("saving sample")
                    # store a sample to listen to
                    ideal = audio.invert_spectrogram(features["stfts"])
                    step = "_" + str(i) + "_"
                    merged = sess.run(tf.summary.merge(
                        [tf.summary.audio("ideal" + step + "\"" + texts + "\"", ideal[None, :], sr),
                         tf.summary.text("text" + step, tf.convert_to_tensor(texts))]
                    ))
                    train_writer.add_summary(merged, i)

if __name__ == '__main__':
    data = "data"
    save_path = "expanse-truncated"
    data_path = os.path.join(data, save_path)
    files = [
        os.path.join(data_path, "data-0.tfrecord"), 
        os.path.join(data_path, "data-1.tfrecord")
    ]
    check_tf_records_with_tensorboard(files, data_path, save_path)