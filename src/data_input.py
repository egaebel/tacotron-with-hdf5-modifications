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

MELS_COL = "mels"
STFTS_COL = "stfts"
TEXTS_COL = "texts"
TEXT_LENS_COL = "text_lens"
SPEECH_LENS_COL = "speech_lens"

BATCH_SIZE = 32
BUFFER_SIZE = 1024
SHUFFLE_BUFFER_SIZE = 10000

MAX_TEXT_LEN = 140

def get_stft_and_mel_std_and_mean(file_name):
    print("Loading stft...")
    stft_file = h5py.File(file_name, 'r')
    stft = stft_file["stfts"]
    print("Loaded stft!")

    # normalize
    # take a sample to avoid memory errors
    index = np.random.randint(len(stft), size=100)

    # h5py only supports indexing by lists with indices in order
    index = sorted(list(set(index.tolist())))

    print("Getting random indexes from stft...")
    indexed_stft = stft[index]
    print("Got random indexes from stft!")

    print("Taking mean and std deviation for stft...")
    stft_mean = np.mean(indexed_stft, axis=(0,1))
    stft_std = np.std(indexed_stft, axis=(0,1), dtype=np.float32)
    print("Got mean and standard deviation for stft!")

    stft_file.close()

    
    print("Loading mel...")
    mel_file = h5py.File(file_name, 'r')
    mel = mel_file["mels"]
    print("Loaded mel!")

    print("Getting random indexes from mel...")
    indexed_mel = mel[index]
    print("Got random indexes from mel!")

    print("Taking mean and std deviation for mel...")
    mel_mean = np.mean(indexed_mel, axis=(0,1))
    mel_std = np.std(indexed_mel, axis=(0,1), dtype=np.float32)
    print("Got mean and standard deviation for mel!")

    mel_file.close()

    return stft_mean, stft_std, mel_mean, mel_std

def build_hdf5_dataset_from_table(file_name, sess, loader, names, shapes, types, ivocab, stft_mean, stft_std, mel_mean, mel_std):
    def tftables_tensor_generator():
        while True:
            loader.q.close()
            loader_results = loader.dequeue()
            return_dict = {name: data_entry for name, data_entry in zip(names, loader_results)}
            
            # Normalize stft
            return_dict['stft'] -= stft_mean
            return_dict['stft'] /= stft_std

            # Normalize mel
            return_dict['mel'] -= mel_mean
            return_dict['mel'] /= mel_std

            tensors_list = [return_dict[name].eval(session=sess) for name in names]
            # tensors_counts = [len(tensors) for tensors in tensors_list]

            yield tuple(tensors_list)

    dataset = tf.data.Dataset.from_generator(tftables_tensor_generator, tuple(types), tuple(shapes))
    # dataset = dataset.repeat()
    iterator = dataset.make_initializable_iterator()
    batch_inputs_from_it = iterator.get_next()
    
    batch_inputs = {na: inp for na, inp in zip(names, batch_inputs_from_it)}

    sess.run(iterator.initializer)#, feed_dict=dict(zip(placeholders, inputs)))
    batch_inputs['stft'] = tf.cast(batch_inputs['stft'], tf.float32)
    batch_inputs['mel'] = tf.cast(batch_inputs['mel'], tf.float32)

    return batch_inputs, stft_mean, stft_std

def build_dataset_with_hdf5_table(file_name):
    inputs = list()
    names = list()
    shapes = list()
    types = list()


    reader = tftables.open_file(filename=file_name, batch_size=BATCH_SIZE)
    table_dict = reader.get_batch(
        path="/data",
        cyclic=True,
        ordered=True)

    # stft
    inputs.append(table_dict[STFTS_COL])
    names.append("stft")
    shapes.append(tf.TensorShape([None, 180, 2050]))
    types.append(tf.float32)

    # mels
    inputs.append(table_dict[MELS_COL])
    names.append("mel")
    shapes.append(tf.TensorShape([None, 180, 160]))
    types.append(tf.float32)

    # texts
    inputs.append(tf.to_int32(table_dict[TEXTS_COL]))
    names.append("text")
    shapes.append(tf.TensorShape([None, table_dict[TEXTS_COL].shape[1]]))
    types.append(tf.int32)

    # text_lens
    inputs.append(tf.to_int32(table_dict[TEXT_LENS_COL]))
    names.append("text_length")
    shapes.append(tf.TensorShape([None]))
    types.append(tf.int32)

    # speech_lens
    inputs.append(tf.to_int32(table_dict[SPEECH_LENS_COL]))
    names.append("speech_length")
    shapes.append(tf.TensorShape([None]))
    types.append(tf.int32)

    print("inputs: %s" % str(inputs))

    loader = reader.get_fifoloader(queue_size=BUFFER_SIZE, inputs=inputs, threads=1)
 
    return loader, reader, names, shapes, types

def get_stft_and_mel_std_and_mean_from_table(file_name):
    print("Loading data...")
    h5py_file = h5py.File(file_name, 'r')
    print("Loaded data!")

    # normalize
    # take a sample to avoid memory errors
    index = np.random.randint(len(h5py_file["data"]["stfts"]), size=100)

    # h5py only supports indexing by lists with indices in order
    index = sorted(list(set(index.tolist())))

    print("Getting random indexes from stft...")
    indexed_stft = h5py_file["data"]["stfts"][index]
    print("Got random indexes from stft!")

    print("Taking mean and std deviation for stft...")
    stft_mean = np.mean(indexed_stft, axis=(0,1))
    stft_std = np.std(indexed_stft, axis=(0,1), dtype=np.float32)
    print("Got mean and standard deviation for stft!")

    print("Getting random indexes from mel...")
    indexed_mel = h5py_file["data"]["mels"][index]
    print("Got random indexes from mel!")

    print("Taking mean and std deviation for mel...")
    mel_mean = np.mean(indexed_mel, axis=(0,1))
    mel_std = np.std(indexed_mel, axis=(0,1), dtype=np.float32)
    print("Got mean and standard deviation for mel!")

    h5py_file.close()
    return stft_mean, stft_std, mel_mean, mel_std

def vocab_prompts_to_string(encoded_prompts, ivocab):
    prompts = []
    for encoded_prompt_list in encoded_prompts:
        prompt = ""
        for encoded_char in encoded_prompt_list:
            prompt += ivocab[encoded_char]
        prompts.append(prompt)
    return prompts

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

