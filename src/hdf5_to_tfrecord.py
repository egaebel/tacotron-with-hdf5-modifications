import h5py
import os
import tensorflow as tf

CHUNK_SIZE = 5000
COL_NAMES = {
	"index": lambda x: tf.train.Feature(int64_list=tf.train.Int64List(value=[x])), 
	"stfts": lambda x: tf.train.Feature(float_list=tf.train.FloatList(value=x.reshape(-1))),
	"mels": lambda x: tf.train.Feature(float_list=tf.train.FloatList(value=x.reshape(-1))),
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
		feature_dict[col_name] = COL_NAMES[col_name](hdf5_row[col_name])
	return tf.train.Features(feature=feature_dict)

def convert_records(file_path):
	dir_name = os.path.dirname(file_path)
	base_file_name = os.path.splitext(file_path)[0]
	tfrecord_file_name_template = "%s-%d.tfrecord"
	tfrecord_file_counter = 0

	hdf5_file = h5py.File(file_path, "r")
	features_list = list()
	index = 0
	while index < hdf5_file[DATASET_NAME].size:
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
	hdf5_file.close()

if __name__ == '__main__':
	convert_records("data/expanse-truncated/data")