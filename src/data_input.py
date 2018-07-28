from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import pickle as pkl
import os
import sys
import io
import tables
import tftables
import h5py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 10000

MAX_TEXT_LEN = 140

def build_dataset_with_hdf5(
        sess, 
        mels_hdf5_file_name,
        stfts_hdf5_file_name,
        texts_hdf5_file_name,
        text_lens_hdf5_file_name,
        speech_lens_hdf5_file_name):
    inputs = list()
    names = list()
    placeholders = list()

    stfts_reader = tftables.open_file(filename=stfts_hdf5_file_name, batch_size=BATCH_SIZE)
    stfts_array_batch_placeholder = stfts_reader.get_batch(
        path="/data",
        ordered=True)
    inputs.append(tf.to_float(stfts_array_batch_placeholder))
    placeholders.append(stfts_array_batch_placeholder)
    names.append("stft")

    mels_reader = tftables.open_file(filename=mels_hdf5_file_name, batch_size=BATCH_SIZE)
    # TODO: Get batch returns a tensor that is the size of the batch size, 
    # other inputs must be manipulated to work well with this
    mels_array_batch_placeholder = mels_reader.get_batch(
        path="/data",
        ordered=True)
    inputs.append(tf.to_float(mels_array_batch_placeholder))
    placeholders.append(mels_array_batch_placeholder)
    names.append("mel")

    # texts
    texts_reader = tftables.open_file(filename=texts_hdf5_file_name, batch_size=BATCH_SIZE)
    texts_array_batch_placeholder = texts_reader.get_batch(
        path="/data",
        ordered=True)
    inputs.append(tf.to_int32(texts_array_batch_placeholder))
    placeholders.append(texts_array_batch_placeholder)
    names.append("text")

    # text_lens
    text_lens_reader = tftables.open_file(filename=text_lens_hdf5_file_name, batch_size=BATCH_SIZE)
    text_lens_array_batch_placeholder = text_lens_reader.get_batch(
        path="/data",
        ordered=True)
    inputs.append(tf.to_int32(text_lens_array_batch_placeholder))
    placeholders.append(text_lens_array_batch_placeholder)
    names.append("text_length")

    # speech_lens
    speech_lens_reader = tftables.open_file(filename=speech_lens_hdf5_file_name, batch_size=BATCH_SIZE)
    speech_lens_array_batch_placeholder = speech_lens_reader.get_batch(
        path="/data",
        ordered=True)
    inputs.append(tf.to_int32(speech_lens_array_batch_placeholder))
    placeholders.append(speech_lens_array_batch_placeholder)
    names.append("speech_length")

    print("Placeholders: %s" % str(placeholders))
    print("inputs: %s" % str(inputs))
    with tf.device('/cpu:0'):
        dataset = tf.data.Dataset.from_tensor_slices(tuple(inputs))
        dataset = dataset.repeat()
        # dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE)
        iterator = dataset.make_initializable_iterator()


        batch_inputs = iterator.get_next()
        batch_inputs = {na: inp for na, inp in zip(names, batch_inputs)}
        for name, inp in batch_inputs.items():
            print(name, inp)

        sess.run(iterator.initializer)

        """
        sess.run(iterator.initializer, feed_dict=dict(zip(placeholders, map(lambda x: np.array(x), inputs))))
        # sess.run(iterator.initializer, feed_dict=dict(zip(placeholders, inputs)))
        batch_inputs['stft'] = tf.cast(batch_inputs['stft'], tf.float32)
        batch_inputs['mel'] = tf.cast(batch_inputs['mel'], tf.float32)
        """

    return batch_inputs

def build_dataset(sess, inputs, names):
    placeholders = []
    for inp in inputs:
        placeholders.append(tf.placeholder(inp.dtype, inp.shape))

    with tf.device('/cpu:0'):
        dataset = tf.contrib.data.Dataset.from_tensor_slices(placeholders)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE)
        iterator = dataset.make_initializable_iterator()

        batch_inputs = iterator.get_next()
        batch_inputs = {na: inp for na, inp in zip(names, batch_inputs)}
        for name, inp in batch_inputs.items():
            print(name, inp)

        sess.run(iterator.initializer, feed_dict=dict(zip(placeholders, inputs)))
        batch_inputs['stft'] = tf.cast(batch_inputs['stft'], tf.float32)
        batch_inputs['mel'] = tf.cast(batch_inputs['mel'], tf.float32)

    return batch_inputs

def load_from_hdf5(dirname):
    # Added comment out
    """
    text = np.load(dirname + 'texts.npy')
    text_length = np.load(dirname + 'text_lens.npy')
    
    print('loading stft')
    stft = np.load(dirname + 'stfts.npy')
    print('loading mel')
    mel = np.load(dirname + 'mels.npy')
    """

    # Added
    print("Loading texts...")
    text_file = h5py.File(dirname + 'texts')
    text = text_file["data"]
    text_length_file = h5py.File(dirname + 'text_lens')
    text_length = text_length_file["data"]

    # Added
    print("Loading stft...")
    stft_file = h5py.File(dirname + "stfts", 'r')
    stft = stft_file["data"]
    print("Loaded stft!")
    print("Loading mel...")
    mel_file = h5py.File(dirname + "mels", 'r')
    mel = mel_file["data"]
    print("Loaded mel!")

    # Added comment out
    """
    speech_length = np.load(dirname + 'speech_lens.npy')
    """

    # Added
    speech_length_file = h5py.File(dirname + "speech_lens", "r")
    speech_length = speech_length_file["data"]

    print('Normalizing...')
    # normalize
    # take a sample to avoid memory errors
    index = np.random.randint(len(stft), size=100)

    # Added, h5py only supports indexing by lists with indices in order
    index = np.sort(index).tolist()

    print("Getting random indexes from stft...")
    indexed_stft = stft[index]
    print("Got random indexes from stft!")
    print("Getting random indexes from mel...")
    indexed_mel = mel[index]
    print("Got random indexes from mel!")

    print("Taking mean and std deviation...")
    stft_mean = np.mean(indexed_stft, axis=(0,1))
    mel_mean = np.mean(indexed_mel, axis=(0,1))
    stft_std = np.std(indexed_stft, axis=(0,1), dtype=np.float32)
    mel_std = np.std(indexed_mel, axis=(0,1), dtype=np.float32)
    print("Got mean and standard deviation!")

    # Added, takes all memory to normalize...
    """
    print("Subtracting mean from stft...")
    stft -= stft_mean
    print("Subtracted mean from stft!")
    print("Dividing stft by standard deviation...")
    stft /= stft_std
    print("Divided stft by standard deviation!")

    print("Subtracting mean from mel...")
    mel -= mel_mean
    print("Subtracted mean from mel!")
    print("Dividing mel by standard deviation...")
    mel /= mel_std
    print("Divided mel by standard deviation!")
    """
    print("Normalized!")

    # Added comment out
    """
    text = np.array(text, dtype=np.int32)
    text_length = np.array(text_length, dtype=np.int32)
    speech_length = np.array(speech_length, dtype=np.int32)
    """

    # Added comment out
    """
    # NOTE: reconstruct zero frames as paper suggests
    speech_length = np.ones(text.shape[0], dtype=np.int32) * mel.shape[1]
    """

    # Added comment out
    """
    inputs = list((text, text_length, speech_length))
    names = ['text', 'text_length', 'speech_length']
    """

    # Added comment out
    """
    num_speakers = 1

    print("Loading speakers...")
    if os.path.exists(dirname + 'speakers.npy'):
        print("Speakers.npy file exists!")
        speakers = np.load(dirname + 'speakers.npy')
        inputs.append(speakers)
        names.append('speaker')
        num_speakers = np.max(speakers) + 1
    print("Loaded %d speakers!" % num_speakers)
    """

    stft_file.close()
    mel_file.close()
    text_file.close()
    text_length_file.close()
    speech_length_file.close()

    return None

def load_from_npy(dirname):
    text = np.load(dirname + 'texts.npy')
    text_length = np.load(dirname + 'text_lens.npy')
    print('loading stft')
    stft = np.load(dirname + 'stfts.npy')
    print('loading mel')
    mel = np.load(dirname + 'mels.npy')
    speech_length = np.load(dirname + 'speech_lens.npy')

    print('normalizing')
    # normalize
    # take a sample to avoid memory errors
    index = np.random.randint(len(stft), size=100)

    stft_mean = np.mean(stft[index], axis=(0,1))
    mel_mean = np.mean(mel[index], axis=(0,1))
    stft_std = np.std(stft[index], axis=(0,1), dtype=np.float32)
    mel_std = np.std(mel[index], axis=(0,1), dtype=np.float32)

    stft -= stft_mean
    mel -= mel_mean
    stft /= stft_std
    mel /= mel_std

    text = np.array(text, dtype=np.int32)
    text_length = np.array(text_length, dtype=np.int32)
    speech_length = np.array(speech_length, dtype=np.int32)

    # NOTE: reconstruct zero frames as paper suggests
    speech_length = np.ones(text.shape[0], dtype=np.int32) * mel.shape[1]

    inputs = list((text, text_length, stft, mel, speech_length))
    names = ['text', 'text_length', 'stft', 'mel', 'speech_length']

    num_speakers = 1

    if os.path.exists(dirname + 'speakers.npy'):
        speakers = np.load(dirname + 'speakers.npy')
        inputs.append(speakers)
        names.append('speaker')
        num_speakers = np.max(speakers) + 1

    return inputs, names, num_speakers

def pad(text, max_len, pad_val):
    return np.array(
        [np.pad(t, (0, max_len - len(t)), 'constant', constant_values=pad_val) for t in text]
    , dtype=np.int32)

def load_prompts(prompts, ivocab):
    vocab = {v: k for k,v in ivocab.items()}
    text = [[vocab[w] for w in p.strip() if w in vocab] for p in prompts]
    text_length = np.array([len(p) for p in prompts])

    # we pad out to a max text length comparable to that used in training
    # this prevents repetition at the end of synthesized prompts
    text = pad(text, MAX_TEXT_LEN, 0)
    
    inputs = tf.train.slice_input_producer([text, text_length], num_epochs=1)
    inputs = {'text': inputs[0], 'text_length': inputs[1]}

    batches = tf.train.batch(inputs,
            batch_size=32,
            allow_smaller_final_batch=True)

    return batches
        
def load_meta(data_path):
    with open('%smeta.pkl' % data_path, 'rb') as vf:
        meta = pkl.load(vf)
    return meta

def generate_attention_plot(alignments):
    plt.imshow(alignments, cmap='hot', interpolation='nearest')
    plt.ylabel('Decoder Steps')
    plt.xlabel('Encoder Steps')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot = tf.image.decode_png(buf.getvalue(), channels=4)
    plot = tf.expand_dims(plot, 0)
    return plot

