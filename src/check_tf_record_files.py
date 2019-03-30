from data_input import count_records
from multiprocessing import Pool

import audio
import data_input
import os
import string
import tensorflow as tf

SAVE_EVERY = 5000

def record_counting_fn(file):
    print("Starting count on file: %s" % file)
    count = 0
    for record in tf.python_io.tf_record_iterator(file):
        count += 1
        if count % 1000 == 0:
            print("On record %d on file: %s" % (count, file))
    print("Finished counting records in: %s" % file)
    return count

def count_records_parallel(files):
    pool = Pool()
    print("Counting number of records from files: %s" % ",".join(files))
    counts = pool.map(record_counting_fn, files)
    pool.close()
        
    total_count = sum(counts)
    print("Counted %d records in %d files!" % (total_count, len(files)))
    return total_count

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
                # "stfts": tf.FixedLenFeature((180, 2050), tf.float32),
                "stfts": tf.FixedLenFeature((504, 2050), tf.float32),
                "stfts_shape": tf.FixedLenFeature((2), tf.int64),
                # "mels": tf.FixedLenFeature((180, 160), tf.float32),
                "mels": tf.FixedLenFeature((504, 160), tf.float32),
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

                    # Debugging THIS script.
                    """
                    print("texts (numbers): %s" % str(texts.eval(session=sess)))
                    for word in texts.eval(session=sess):
                        try:
                            print("word: %s" % str(ivocab[word]))
                        except:
                            print("invalid word: %s" % str(word))
                    print("ivocab: %s" % str(ivocab))
                    """

                    # Convert integers to words
                    texts = "".join(filter(lambda x: x != "<pad>", [ivocab[word] for word in texts.eval(session=sess)]))
                    print("Texts: %s" % texts)
                    texts_filtered = "".join(filter(lambda x: x in set(string.printable), texts))
                    # print("Texts filtered: %s" % texts_filtered)

                    print("stfts shape: %s" % str(features["stfts"].shape))
                    print("mels shape: %s" % str(features["mels"].shape))

                    print("saving sample")
                    # store a sample to listen to
                    ideal = audio.invert_spectrogram(features["stfts"])
                    step = "_" + str(i) + "_"
                    merged = sess.run(tf.summary.merge(
                        [tf.summary.audio("ideal" + step + "\"" + texts_filtered + "\"", ideal[None, :], sr),
                         tf.summary.text("text" + step, tf.convert_to_tensor(texts_filtered))]
                    ))
                    train_writer.add_summary(merged, i)

if __name__ == '__main__':
    data = "data"
    # save_path = "expanse-truncated"
    save_path = "expanse"
    data_path = os.path.join(data, save_path)
    num_tf_record_files = 21
    files = [os.path.join(data_path, "data-%d.tfrecord" % num) for num in range(num_tf_record_files)]
    """
    files = [
        os.path.join(data_path, "data-15.tfrecord"),
        os.path.join(data_path, "data-16.tfrecord"),
    ]
    #"""
    # check_tf_records_with_tensorboard(files, data_path, save_path)
    count_records_parallel(files)