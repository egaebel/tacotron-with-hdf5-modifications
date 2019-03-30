
import h5py
import os
import tensorflow as tf

CHUNK_SIZE = 5000
COL_NAMES = {
    "index": lambda x: tf.train.Feature(int64_list=tf.train.Int64List(value=[x])),
    "stfts": lambda x: tf.train.Feature(float_list=tf.train.FloatList(value=x.reshape(-1))),
    "stfts_shape": lambda x: tf.train.Feature(int64_list=tf.train.Int64List(value=list(x.shape))),
    "mels": lambda x: tf.train.Feature(float_list=tf.train.FloatList(value=x.reshape(-1))),
    "mels_shape": lambda x: tf.train.Feature(int64_list=tf.train.Int64List(value=list(x.shape))),
    "texts": lambda x: tf.train.Feature(int64_list=tf.train.Int64List(value=x)),
    "text_lens": lambda x: tf.train.Feature(int64_list=tf.train.Int64List(value=[x])),
    "speech_lens": lambda x: tf.train.Feature(int64_list=tf.train.Int64List(value=[x])),
}
DATASET_NAME = "data"

def write_tfrecords(file_path, features_list):
    with tf.python_io.TFRecordWriter(file_path) as writer:
        for features in features_list:
            writer.write(tf.train.Example(features=features).SerializeToString())

def hdf5_row_to_features(hdf5_row):
    feature_dict = dict()
    for col_name in COL_NAMES.keys():
        if col_name == "stfts_shape" or col_name == "mels_shape":
            continue
        if col_name == "index" and hdf5_row["index"] % 100 == 0:
            print("index: %d" % hdf5_row["index"])
        feature_dict[col_name] = COL_NAMES[col_name](hdf5_row[col_name])
        if col_name == "stfts":
            feature_dict["stfts_shape"] = COL_NAMES["stfts_shape"](hdf5_row[col_name])
        if col_name == "mels":
            feature_dict["mels_shape"] = COL_NAMES["mels_shape"](hdf5_row[col_name])
    return tf.train.Features(feature=feature_dict)

def convert_records(file_path):
    dir_name = os.path.dirname(file_path)
    base_file_name = os.path.splitext(file_path)[0]
    tfrecord_file_name_template = "%s-%d.tfrecord"
    tfrecord_file_counter = 0

    hdf5_file = h5py.File(file_path, "r")
    features_list = list()
    index = 0
    print("Dataset size: %d" % hdf5_file[DATASET_NAME].size)
    while index < hdf5_file[DATASET_NAME].size:
        if index % 100 == 0:
            print("iteration index: %d" % index)
        features = hdf5_row_to_features(hdf5_file[DATASET_NAME][index])
        features_list.append(features)

        # Write chunk to file.
        if index % CHUNK_SIZE == 0 and index != 0:
            write_tfrecords(
                os.path.join(
                    # dir_name,
                    tfrecord_file_name_template % (base_file_name, tfrecord_file_counter)),
                features_list)
            tfrecord_file_counter += 1
            features_list = list()
        index += 1

    # Write remainder to file.
    if index % CHUNK_SIZE != 0:
        write_tfrecords(
            os.path.join(
                # dir_name,
                tfrecord_file_name_template % (base_file_name, tfrecord_file_counter)),
            features_list)

    print("Dataset size: %d" % hdf5_file[DATASET_NAME].size)
    hdf5_file.close()

def inspect_tf_record_file(file_path, result_chunking=1):
    count = 0
    for example in tf.python_io.tf_record_iterator(file_path):
        result = tf.train.Example.FromString(example)
        if count % result_chunking == 0:
            print("result: %s" % result)
        count += 1
    print("Total count: %d" % count)


if __name__ == '__main__':
    convert_records("data/expanse-truncated/data")
    # convert_records("data/expanse/data")
    # convert_records("data/expanse-error/data")
    print("Done!")

    # inspect_tf_record_file("data/expanse-truncated/data-0.tfrecord", 10000)