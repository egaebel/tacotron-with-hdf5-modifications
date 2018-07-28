from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import glob
import pickle as pkl
from tqdm import tqdm
import sys
import string
import tables

import audio
import argparse

mini = False
DATA_DIR = 'data/'

vocab = {}
ivocab = {}
vocab['<pad>'] = 0
ivocab[0] = '<pad>'

""" 
    To add a new dataset, write a new prepare function such as those below
    This needs to return a dictionary containing 
    the following two lists of strings:

        'prompts' -- a list of strings of the text for each example 
        'audio_files' -- a corresponding list of audio filenames (preferably wav)

    Finally, add the function to the 'prepare_functions' dictionary below

    Examples for the ARCTIC, Nancy and VCTK datasets are shown below

    Note: For multi-speaker datasets, one can also add a 'speakers'
    object to the dictionary
    This contains the speaker id (an int) for each utterance
"""
def prepare_expanse():
    expanse_root_dir = "../../../audiobook-dataset-creator/src/expanse-data/"
    audio_files_dir = os.path.join(expanse_root_dir, "all-expanse-audio")
    prompts_dir = os.path.join(expanse_root_dir, "all-expanse-sentences")
    prompts = list()
    for prompt_file_name in sorted(os.listdir(prompts_dir)):
        with open(os.path.join(prompts_dir, prompt_file_name)) as prompt_file:
            prompts.append(prompt_file.read())
    audio_files = [
        os.path.join(audio_files_dir, audio_file_name)
        for audio_file_name in sorted(os.listdir(audio_files_dir))]

    return {'prompts': prompts, 'audio_files': audio_files}

def prepare_arctic():
    proto_file = DATA_DIR + 'arctic/train.proto'
    txt_file = DATA_DIR + 'arctic/etc/arctic.data'

    prompts = []
    audio_files = []

    with open(txt_file, 'r') as tff:
        for line in tff:
            spl = line.split()
            id = spl[1]
            text = ' '.join(spl[2:-1])
            text = text[1:-1]

            audio_file = DATA_DIR + 'arctic/wav/{}.wav'.format(id)

            prompts.append(text)
            audio_files.append(audio_file)

    return {'prompts': prompts, 'audio_files': audio_files}

def prepare_nancy():
    nancy_dir = DATA_DIR + 'nancy/' 
    txt_file = nancy_dir + 'prompts.data'

    prompts = []
    audio_files = []

    with open(txt_file, 'r') as ttf:
        for line in ttf:
            id = line.split()[1]
            text = line[line.find('"')+1:line.rfind('"')-1]

            audio_file = nancy_dir + 'wavn/' + id + '.wav'

            prompts.append(text)
            audio_files.append(audio_file)

    return {'prompts': prompts, 'audio_files': audio_files}

def prepare_vctk():
    # adapted from https://github.com/buriburisuri/speech-to-text-wavenet/blob/master/preprocess.py
    import pandas as pd

    prompts = []
    audio_files = []
    speakers = []

    # read label-info
    df = pd.read_table(DATA_DIR + 'vctk/speaker-info.txt', usecols=['ID'],
                       index_col=False, delim_whitespace=True)

    # assign speaker IDs
    speaker_ids = {str(uid): i for i, uid in enumerate(df.ID.values)}
    print(speaker_ids)
    

    # read file IDs
    file_ids = []
    for d in [DATA_DIR + 'vctk/txt/p%d/' % uid for uid in df.ID.values]:
        file_ids.extend([f[-12:-4] for f in sorted(glob.glob(d + '*.txt'))])

    for i, f in tqdm(enumerate(file_ids), total=len(file_ids)):

        # wave file name
        audio_file = DATA_DIR + 'vctk/wav48/%s/' % f[:4] + f + '.wav'
        txt_file = DATA_DIR + 'vctk/txt/%s/' % f[:4] + f + '.txt'

        with open(txt_file, 'r') as tff:
            text = tff.read().strip()

        prompts.append(text)
        audio_files.append(audio_file)
        
        speakers.append(speaker_ids[f[1:4]])

    return {'prompts': prompts, 'audio_files': audio_files, 'speakers': speakers}

# Add new data preparation functions here
prepare_functions = {
        'arctic': prepare_arctic,
        'expanse': prepare_expanse,
        'nancy': prepare_nancy,
        'vctk': prepare_vctk
}

###########################################################################
# Below functions should not need to be altered when adding a new dataset #
###########################################################################

def process_char(char):
    if not char in vocab:
        next_index = len(vocab)
        vocab[char] = next_index
        ivocab[next_index] = char
    return vocab[char]

def pad_to_dense(inputs):
    max_len = max(r.shape[0] for r in inputs)
    if len(inputs[0].shape) == 1:
        padded = [np.pad(inp, (0, max_len - inp.shape[0]), 'constant', constant_values=0) \
                        for i, inp in enumerate(inputs)]
    else:
        padded = [np.pad(inp, ((0, max_len - inp.shape[0]),(0,0)), 'constant', constant_values=0) \
                        for i, inp in enumerate(inputs)]
    padded = np.stack(padded)
    return padded

def save_to_npy(texts, text_lens, mels, stfts, speech_lens, filename):
    texts = pad_to_dense(texts)

    text_lens, speech_lens = np.array(text_lens), np.array(speech_lens)

    inputs = texts, text_lens, mels, stfts, speech_lens
    names = 'texts', 'text_lens', 'mels', 'stfts', 'speech_lens'
    names = ['data/%s/%s' % (filename, name) for name in names]

    for name, inp in zip(names, inputs):
        print(name, inp.shape)
        np.save(name, inp, allow_pickle=False)

def create_hdf5_files(max_freq_length, filename):
    short_names = 'mels', 'stfts'
    names = ['data/%s/%s' % (filename, name) for name in short_names]

    atoms = [tables.Float16Atom(), tables.Float16Atom()]
    sizes = [
        (0, max_freq_length, 80 * audio.r),
        (0, max_freq_length, 1025 * audio.r)]
    for short_name, name, atom, size in zip(short_names, names, atoms, sizes):
        try:
            os.remove(name)
        except:
            pass
        tables_file = tables.open_file(name, mode='w')
        tables_file.create_earray(tables_file.root, "data", atom, size)
        tables_file.close()

def append_to_hdf5(mels, stfts, filename):
    inputs = mels, stfts
    short_names = 'mels', 'stfts'
    names = ['data/%s/%s' % (filename, name) for name in short_names]
    atoms = [tables.Float16Atom(), tables.Float16Atom()]

    for short_name, name, inp, atom in zip(short_names, names, inputs, atoms):
        tables_file = tables.open_file(name, mode='a')
        tables_file.root.data.append(inp)
        tables_file.close()

def save_to_hdf5(array, filename):
    """
    inputs = texts, text_lens, speech_lens
    names = 'texts', 'text_lens', 'speech_lens'
    names = ['data/%s/%s' % (filename, name) for name in names]

    for name, inp in zip(names, inputs):
        print(name, inp.shape)
        np.save(name, inp, allow_pickle=False)
    """
    tables_file = tables.open_file(filename, mode='w')
    carray = tables_file.create_carray(
        tables_file.root,
        "data",
        tables.Atom.from_dtype(array.dtype),
        array.shape)
    carray[:] = array
    tables_file.close()

def save_vocab(name, sr=16000):
    global vocab
    global ivocab
    print('saving vocab')
    with open('data/%s/meta.pkl' % name, 'wb') as vf:
        pkl.dump({'vocab': ivocab, 'r': audio.r, 'sr': sr}, vf, protocol=2)

def preprocess(data, data_name, sr=16000):
    # Added
    example_batch_size = 2 # 5000

    # get count of examples from text file
    num_examples = len(data['prompts'])

    # pad out all these jagged arrays and store them in an npy file
    texts = []
    text_lens = []
    speech_lens = []

    max_freq_length = audio.maximum_audio_length // (audio.r*audio.hop_length)
    print("num_examples: %s" % str(num_examples))
    print("max_freq_length: %s" % str(max_freq_length))
    print("1025*audio.r: %s" % str(1025*audio.r))
    # Added
    mels = np.zeros((example_batch_size, max_freq_length, 80*audio.r), dtype=np.float16)
    stfts = np.zeros((example_batch_size, max_freq_length, 1025*audio.r), dtype=np.float16)
    """
    stfts = np.zeros((num_examples, max_freq_length, 1025*audio.r), dtype=np.float16)
    mels = np.zeros((num_examples, max_freq_length, 80*audio.r), dtype=np.float16)
    """

    # Added
    create_hdf5_files(max_freq_length, data_name)

    count = 0
    for text, audio_file in tqdm(
            zip(data['prompts'], data['audio_files']), total=num_examples):

        text = [process_char(c) for c in list(text)]
        mel, stft = audio.process_audio(audio_file, sr=sr)

        if mel is not None:
            texts.append(np.array(text))
            text_lens.append(len(text))
            speech_lens.append(mel.shape[0])

            # Added
            mels[count % example_batch_size] = mel
            stfts[count % example_batch_size] = stft


            # Added comment out
            # mels[count] = mel
            # stfts[count] = stft

            count += 1

            # Added
            if count % example_batch_size == 0:
                append_to_hdf5(mels, stfts, data_name)


    # Added
    append_to_hdf5(mels[:count % example_batch_size], stfts[:count % example_batch_size], data_name)

    # Added comment out
    """
    mels = mels[:len(texts)]
    stfts = stfts[:len(texts)]
    """

    # Added comment out
    """
    save_to_npy(texts, text_lens, mels, stfts, speech_lens, data_name)
    """

    # Added
    inputs = pad_to_dense(texts), np.array(text_lens), np.array(speech_lens)
    short_names = 'texts', 'text_lens', 'speech_lens'
    filenames = ['data/%s/%s' % (data_name, name) for name in short_names]
    for filename, inp in zip(filenames, inputs):
        print("Saving to hdf5: %s, %s" % (filename, inp.shape))
        print("input: %s" % inp)
        save_to_hdf5(inp, filename)

    if 'speakers' in data:
        np.save('data/%s/speakers.npy' % data_name, data['speakers'], allow_pickle=False)

    # save vocabulary
    save_vocab(data_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='specify the name of the dataset to preprocess')
    args = parser.parse_args()
    
    if args.dataset not in prepare_functions:
        raise NotImplementedError('No prepare function exists for the %s dataset' % args.dataset)
    sr = 24000 if args.dataset == 'vctk' else 16000
    data = prepare_functions[args.dataset]()
    preprocess(data, args.dataset, sr=sr)

