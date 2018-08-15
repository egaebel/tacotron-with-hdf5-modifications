from data_input import get_stft_and_mel_std_and_mean

import itertools
import numpy as np
import tensorflow as tf
import tftables

BATCH_SIZE = 2
BUFFER_SIZE = 1000
ITERATIONS = 8

def build_dataset_with_hdf5(file_name):
    inputs = list()
    names = list()
    shapes = list()
    types = list()
    placeholders = list()

    with tf.device('/cpu:0'):
        reader = tftables.open_file(filename=file_name, batch_size=BATCH_SIZE)
        stfts_array_batch_placeholder = reader.get_batch(
            path="/stfts",
            ordered=True)
        # inputs.append(tf.to_float(stfts_array_batch_placeholder))
        inputs.append(stfts_array_batch_placeholder)
        placeholders.append(stfts_array_batch_placeholder)
        names.append("stft")
        shapes.append(tf.TensorShape([None, 180, 2050]))
        types.append(tf.float32)

        mels_array_batch_placeholder = reader.get_batch(
            path="/mels",
            ordered=True)
        # inputs.append(tf.to_float(mels_array_batch_placeholder))
        inputs.append(mels_array_batch_placeholder)
        placeholders.append(mels_array_batch_placeholder)
        names.append("mel")
        shapes.append(tf.TensorShape([None, 180, 160]))
        types.append(tf.float32)

        # texts
        texts_array_batch_placeholder = reader.get_batch(
            path="/texts",
            ordered=True)
        inputs.append(tf.to_int32(texts_array_batch_placeholder))
        placeholders.append(texts_array_batch_placeholder)
        names.append("text")
        shapes.append(tf.TensorShape([None, 164]))
        types.append(tf.int32)

        # text_lens
        text_lens_array_batch_placeholder = reader.get_batch(
            path="/text_lens",
            ordered=True)
        inputs.append(tf.to_int32(text_lens_array_batch_placeholder))
        placeholders.append(text_lens_array_batch_placeholder)
        names.append("text_length")
        shapes.append(tf.TensorShape([None]))
        types.append(tf.int32)

        # speech_lens
        speech_lens_array_batch_placeholder = reader.get_batch(
            path="/speech_lens",
            ordered=True)
        inputs.append(tf.to_int32(speech_lens_array_batch_placeholder))
        placeholders.append(speech_lens_array_batch_placeholder)
        names.append("speech_length")
        shapes.append(tf.TensorShape([None]))
        types.append(tf.int32)

        print("Placeholders: %s" % str(placeholders))
        print("inputs: %s" % str(inputs))

        loader = reader.get_fifoloader(queue_size=BUFFER_SIZE, inputs=inputs, threads=1)

    return loader, reader, names, shapes, types

def test_from_generator():
    def gen():
        for i in itertools.count(5):
            yield (np.ones((20, 20)), i, np.zeros((10, 10)))

    dataset = tf.data.Dataset.from_generator(gen, (tf.int64, tf.int64, tf.int64), (tf.TensorShape([20, 20]), tf.TensorShape([]), tf.TensorShape([10, 10])))
    value = dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        output = sess.run(value)
        print("output: %s" % str(output))

def test_from_generator_with_tftables():
    file_name = "data/expanse/data"

    loader, reader, names, shapes, types = build_dataset_with_hdf5(file_name)

    print("Obtaining means and stds....")
    stft_mean, stft_std, mel_mean, mel_std = get_stft_and_mel_std_and_mean(file_name)
    print("Obtained means and stds!")

    with tf.Session() as sess:
        def tftables_tensor_generator():
            # for i in itertools.count(ITERATIONS // BATCH_SIZE):
            while True:
                loader_results = loader.dequeue()

                print("loader_results length: %d" % len(loader_results))
                print("loader_results[0]: %s" % loader_results)
                print("loader_results[1]: %s" % loader_results)
                print("loader_results[2]: %s" % loader_results)
                print("loader_results[3]: %s" % loader_results)
                print("loader_results[4]: %s" % loader_results)

                return_dict = {name: data_entry for name, data_entry in zip(names, loader_results)}
                
                print("Applying normalization to stft and mel.....")
                return_dict['stft'] -= stft_mean
                return_dict['stft'] /= stft_std

                return_dict['mel'] -= mel_mean
                return_dict['mel'] /= mel_std
                print("Applied normalization to stft and mel!")

                print("Yielding: %s" % str([return_dict[name] for name in names]))
                yield tuple([return_dict[name].eval(session=sess) for name in names])

        # dataset = tf.contrib.data.Dataset.from_tensor_slices(inputs)
        # print("Inputs: %s" % str(inputs))
        print("Types: %s" % types)
        print("Shapes: %s" % shapes)
        dataset = tf.data.Dataset.from_generator(tftables_tensor_generator, tuple(types), tuple(shapes))
        value = dataset.make_one_shot_iterator().get_next()

    expanse_outputs = list()
    with tf.Session() as sess:
        with loader.begin(sess):
            for _ in range(ITERATIONS // BATCH_SIZE):
                output = sess.run(value)
                for i in range(BATCH_SIZE):
                    
                    # print("output: %s" % str(output))
                    # print("output[0] len: %d" % len(output[0].tolist()))
                    print("output[0] sample: %s" % str(output[0][i]))
                    print("output[1] sample: %s" % str(output[1][i]))
                    # [expanse_outputs.append(output)

    reader.close()

    print("\n\n\n\n\n")
    print("================================================================================")
    print("T  H  E    E  X  P  A  N  S  E    B  Y    J  A  M  E  S    S  A    C  O  R  E  Y")
    print("================================================================================")
    """
    print("\n\n\n\n\n")
    print("0: %s" % str(expanse_outputs[0]))
    print("\n\n\n\n\n")    
    print("31: %s" % str(expanse_outputs[1]))
    print("\n\n\n\n\n")    
    print("1: %s" % str(expanse_outputs[0]))
    print("\n\n\n\n\n")    
    print("32: %s" % str(expanse_outputs[0]))
    print("\n\n\n\n\n")    
    """

if __name__ == '__main__':
    # test_from_generator()
    test_from_generator_with_tftables()
