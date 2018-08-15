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
BUFFER_SIZE = 1024
SHUFFLE_BUFFER_SIZE = 10000

MAX_TEXT_LEN = 140

# def build_hdf5_dataset(file_name, sess, input_batch_inputs, loader):
def build_hdf5_dataset(file_name, sess, loader, names, shapes, types, ivocab):
    """
    names = input_batch_inputs.keys()
    inputs = [input_batch_inputs[name] for name in names]
    placeholders = []
    types = []
    shapes = []
    for inp in inputs:
        placeholders.append(tf.placeholder(inp.dtype, inp.shape))
        types.append(inp.dtype)
        shapes.append(inp.shape)
    """
    """
    types = [tf.float16, tf.int32, tf.float16, tf.int32, tf.int32]
    shapes = [(180, 2050), (164), (180, 160), (), ()]
    names = ["stft", "text", "mel", "text_length", "speech_length"]
    """

    print("Obtaining means and stds....")
    stft_mean, stft_std, mel_mean, mel_std = get_stft_and_mel_std_and_mean(file_name)
    print("Obtained means and stds!")

    # with tf.device('/cpu:0'):
    def tftables_tensor_generator():
        while True:
            print("Queue Size: %s" % str(loader.q.size().eval(session=sess)))
            print("is_closd: %s" % str(loader.q.is_closed().eval(session=sess)))
            loader.q.close()
            loader_results = loader.dequeue()
            print("loader_results: %s" % str(loader_results))
            print("Queue Size: %s" % str(loader.q.size().eval(session=sess)))
            print("is_closed: %s" % str(loader.q.is_closed().eval(session=sess)))
            return_dict = {name: data_entry for name, data_entry in zip(names, loader_results)}
            
            # Normalize stft
            return_dict['stft'] -= stft_mean
            return_dict['stft'] /= stft_std

            # Normalize mel
            return_dict['mel'] -= mel_mean
            return_dict['mel'] /= mel_std

            tensors_list = [return_dict[name].eval(session=sess) for name in names]
            tensors_counts = [len(tensors) for tensors in tensors_list]

            texts_index = 0
            for name in names:
                if name == "text":
                    break
                texts_index += 1
            """
            # print("tensors_list:\n%s\n" % str(tensors_list))
            print("tensor counts:\n%s\n" % str(tensors_counts))
            # print("texts in list:\n%s\n" % str(return_dict["text"].eval(session=sess).tolist()))
            print("Loader state:")
            print("dtypes: %s" % str(loader.q.dtypes))
            print("Shapes: %s" % str(loader.q.shapes))
            """
            print("Texts:\n%s\n" % str(tensors_list[texts_index].tolist()))
            print("Texts:\n%s\n" % str("\n".join(vocab_prompts_to_string(tensors_list[texts_index].tolist(), ivocab))))

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

def build_dataset_with_hdf5(file_name):
    inputs = list()
    names = list()
    shapes = list()
    types = list()


    reader = tftables.open_file(filename=file_name, batch_size=BATCH_SIZE)
    stfts_array_batch_placeholder = reader.get_batch(
        path="/stfts",
        cyclic=True,
        ordered=False)
    inputs.append(stfts_array_batch_placeholder)
    names.append("stft")
    shapes.append(tf.TensorShape([None, 180, 2050]))
    types.append(tf.float32)

    mels_array_batch_placeholder = reader.get_batch(
        path="/mels",
        cyclic=True,
        ordered=False)
    inputs.append(mels_array_batch_placeholder)
    names.append("mel")
    shapes.append(tf.TensorShape([None, 180, 160]))
    types.append(tf.float32)

    # texts
    texts_array_batch_placeholder = reader.get_batch(
        path="/texts",
        cyclic=True,
        ordered=False)
    inputs.append(tf.to_int32(texts_array_batch_placeholder))
    names.append("text")
    shapes.append(tf.TensorShape([None, texts_array_batch_placeholder.shape[1]]))
    types.append(tf.int32)

    # text_lens
    text_lens_array_batch_placeholder = reader.get_batch(
        path="/text_lens",
        cyclic=True,
        ordered=False)
    inputs.append(tf.to_int32(text_lens_array_batch_placeholder))
    names.append("text_length")
    shapes.append(tf.TensorShape([None]))
    types.append(tf.int32)

    # speech_lens
    speech_lens_array_batch_placeholder = reader.get_batch(
        path="/speech_lens",
        cyclic=True,
        ordered=False)
    inputs.append(tf.to_int32(speech_lens_array_batch_placeholder))
    names.append("speech_length")
    shapes.append(tf.TensorShape([None]))
    types.append(tf.int32)

    print("inputs: %s" % str(inputs))

    loader = reader.get_fifoloader(queue_size=BUFFER_SIZE, inputs=inputs, threads=1)
 
    return loader, reader, names, shapes, types

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

