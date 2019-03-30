import tensorflow as tf
import numpy as np
import sys
import os
import re
import data_input
import librosa

from tqdm import tqdm
import argparse

import audio

SAVE_EVERY = 1000
RESTORE_FROM = None

ANNEALING_STEPS = 500000

# TODO: Add caching metadata for mean and std and add metadata file to tf record dir to indicate number of records, etc

def train(model, config, num_steps=1000000):

    sr = 24000 if 'vctk' in config.data_path else 16000
    meta = data_input.load_meta(config.data_path)
    config.r = meta['r']
    ivocab = meta['vocab']
    config.vocab_size = len(ivocab)

    print("Sampling mean and std...")
    if args.hdf5:
        stft_mean, stft_std, mel_mean, mel_std = data_input.get_stft_and_mel_std_and_mean_from_table(
            os.path.join(config.data_path, "data"))
    else:
        stft_mean, stft_std, mel_mean, mel_std = data_input.get_stft_and_mel_std_and_mean_from_tfrecords(
            config.tf_record_files)
    print("Sampled mean and std!")

    
    print("Building dataset...")
    loader, reader, names, shapes, types = data_input.build_dataset_with_hdf5_table(
        os.path.join(config.data_path, "data"))
    print("Built dataset!")
    

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Session(config=config_proto) as sess:
        if args.hdf5:
            batch_inputs, stft_mean, stft_std = data_input.build_hdf5_dataset_from_table(
                os.path.join(config.data_path, "data"), 
                sess, 
                loader, 
                names, 
                shapes, 
                types, 
                ivocab,
                stft_mean,
                stft_std,
                mel_mean,
                mel_std)
        else:
            batch_inputs = data_input.build_tfrecord_dataset(
                config.tf_record_files,
                sess,
                names,
                ivocab,
                stft_mean,
                stft_std,
                mel_mean,
                mel_std)

        tf.Variable(stft_mean, name='stft_mean')
        tf.Variable(stft_std, name='stft_std')

        print("Initializing model...")
        # initialize model
        model = model(config, batch_inputs, train=True)
        print("Model initialized!")

        train_writer = tf.summary.FileWriter('log/' + config.save_path + '/train', sess.graph)

        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        print("Starting queue runners...")
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print("Started queue runners!")

        saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=3)

        if config.restore:
            print('restoring weights')
            latest_ckpt = tf.train.latest_checkpoint(
                'weights/' + config.save_path[:config.save_path.rfind('/')]
            )
            if RESTORE_FROM is None:
                if latest_ckpt is not None:
                    saver.restore(sess, latest_ckpt)
            else:
                saver.restore(sess, 'weights/' + config.save_path + '-' + str(RESTORE_FROM))

        lr = model.config.init_lr
        annealing_rate = model.config.annealing_rate
        if config.restore:
            print("Restored global step: %s" % str(model.global_step.eval(sess)))
            lr *= (annealing_rate**(model.global_step.eval(sess) // ANNEALING_STEPS))
            print("Recovered learning rate: '%s'" % str(lr))
        print("Using learning rate: '%s' and annealing rate: '%s'" % (lr, annealing_rate))
        
        print("Looping over num_steps: %s" % str(num_steps))
        with loader.begin(sess):
            for _ in tqdm(range(num_steps)):
                print("Running sess...")
                out = sess.run([
                    model.train_op,
                    model.global_step,
                    model.loss,
                    model.output,
                    model.alignments,
                    model.merged,
                    batch_inputs
                    ], feed_dict={model.lr: lr})
                _, global_step, loss, output, alignments, summary, inputs = out
                print("Finished run: %d!" % global_step)

                train_writer.add_summary(summary, global_step)

                # detect gradient explosion
                if loss > 1e9 and global_step > 50000:
                    print('loss exploded')
                    break

                if global_step % ANNEALING_STEPS == 0:
                    old_lr = lr
                    lr *= annealing_rate
                    print("Updated learning rate from: %s to %s" % (str(old_lr), str(lr)))

                if global_step % SAVE_EVERY == 0 and global_step != 0:

                    print('saving weights')
                    if not os.path.exists('weights/' + config.save_path):
                        os.makedirs('weights/' + config.save_path)
                    saver.save(sess, 'weights/' + config.save_path, global_step=global_step)

                    print('saving sample')
                    print("stft shape: %s" % str(inputs['stft'][0].shape))
                    # store a sample to listen to
                    ideal = audio.invert_spectrogram(inputs['stft'][0] * stft_std + stft_mean)
                    sample = audio.invert_spectrogram(output[0] * stft_std + stft_mean)
                    attention_plot = data_input.generate_attention_plot(alignments[0])
                    step = '_' + str(global_step) + '_'
                    # Remove pad words
                    text_string = texts = "".join(
                        filter(
                            lambda x: x != "<pad>", 
                            [ivocab[word] for word in inputs['text'][0]]))
                    # Remove unicode chars, replacing them with 0
                    text_string = "".join(
                        map(
                            lambda x: "0" 
                                # This is the REGEX specified in name_scope in ops.py in tensorflow
                                if re.match("[A-Za-z0-9_.\\-/ ]", x) is None 
                                else x, 
                            text_string))
                    text_string = text_string.strip()
                    quoted_text_string = "\"" + text_string + "\""
                    print("ideal: %s %s %s" % (str(step), str(ideal[None, :]), str(sr)))
                    print("sample: %s %s %s" % (str(step), str(sample[None, :]), str(sr)))
                    merged = sess.run(tf.summary.merge(
                        [tf.summary.audio('ideal' + step + text_string, ideal[None, :], sr),
                         tf.summary.audio('sample' + step + text_string, sample[None, :], sr),
                         tf.summary.image('attention' + step, attention_plot),
                         tf.summary.text('text' + step, tf.convert_to_tensor(quoted_text_string))]
                    ))
                    train_writer.add_summary(merged, global_step)
                if global_step % 50 == 0:
                    print("This is reassurance. Global step at: %d" % global_step)

            coord.request_stop()
            coord.join(threads)
        reader.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train-set', default='nancy')
    parser.add_argument('-d', '--debug', type=bool, default=False)
    parser.add_argument('-r', '--restore', type=bool, default=False)
    parser.add_argument('-f', '--hdf5', type=bool, default=False)
    args = parser.parse_args()

    from models.tacotron import Tacotron, Config
    model = Tacotron
    config = Config()
    config.data_path = 'data/%s/' % args.train_set
    tf_record_files = list()
    for file_name in os.listdir(config.data_path):
        if os.path.splitext(file_name)[1] == ".tfrecord":
            tf_record_files.append(os.path.join(config.data_path, file_name))
    print("Read .tfrecord files:\n%s" % "\n".join(tf_record_files))
    config.tf_record_files = tf_record_files

    config.restore = args.restore
    if args.debug: 
        config.save_path = 'debug'
    else:
        config.save_path = '%s/tacotron' % args.train_set
    print('Buliding Tacotron')

    train(model, config)
