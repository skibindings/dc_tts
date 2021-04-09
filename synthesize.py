# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

import os

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph
from utils import *
from data_load import load_data
from scipy.io.wavfile import write
from tqdm import tqdm
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
import keras.losses

import matplotlib.pyplot as plt
import librosa.display

import pandas as pd
import librosa

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def GLU(x):
    return K.sigmoid(x) * x
	
def custom_loss(y_true, y_pred): 
    loss = K.cos(y_true - y_pred)
    loss = -K.sum(loss, axis=1)

    return loss

def synthesize():
    # Load data
    L = load_data("synthesize")

    # Load graph
    g = Graph(mode="synthesize"); print("Graph loaded")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore parameters
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        saver1 = tf.train.Saver(var_list=var_list)
        saver1.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-1"))
        print("Text2Mel Restored!")

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        saver2 = tf.train.Saver(var_list=var_list)
        saver2.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-2"))
        print("SSRN Restored!")

        # Feed Forward
        ## mel
        Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
        prev_max_attentions = np.zeros((len(L),), np.int32)
        for j in tqdm(range(hp.max_T)):
            _gs, _Y, _max_attentions, _alignments = \
                sess.run([g.global_step, g.Y, g.max_attentions, g.alignments],
                         {g.L: L,
                          g.mels: Y,
                          g.prev_max_attentions: prev_max_attentions})
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]

        Y_pre = Y[0].T
        idx = np.argwhere(np.all(Y_pre[..., :] < 0.1, axis=0))
        Y_pre = np.delete(Y_pre, idx, axis=1)
		
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        p = librosa.display.specshow(Y_pre, hop_length=hp.hop_length, ax=ax, y_axis='mel', x_axis='time')
        fig.savefig(hp.sampledir + '/mel_spec.png')

        # Get magnitude
        Z = sess.run(g.Z, {g.Y: Y})
		
        Z_pre = Z[0].T
        idx = np.argwhere(np.all(Z_pre[..., :] < 0.1, axis=0))
        Z_pre = np.delete(Z_pre, idx, axis=1)
		
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        p = librosa.display.specshow(Z_pre, hop_length=hp.hop_length, ax=ax, y_axis='log', x_axis='time')
        fig.savefig(hp.sampledir + '/mag_spec.png')

        model = None
        if hp.phase_reconstruction == True:
            #get_custom_objects().update({'GLU': })
            keras.losses.custom_loss = custom_loss
            model = keras.models.load_model(hp.phasemodeldir,custom_objects={'GLU' : Activation(GLU)})# 'loss': custom_loss, 
        
        # Generate wav files
        if not os.path.exists(hp.sampledir): os.makedirs(hp.sampledir)
        for i, mag in enumerate(Z):
            print("Working on file", i+1)
            wav = spectrogram2wav(mag,model)
            write(hp.sampledir + "/{}.wav".format(i+1), hp.sr, wav)

if __name__ == '__main__':
    synthesize()
    print("Done")


