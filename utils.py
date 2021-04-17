# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''
from __future__ import print_function, division

import numpy as np
import librosa
import math
import os, copy
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy import signal

from hyperparams import Hyperparams as hp
import tensorflow as tf

def get_spectrograms(fpath):
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.

    Args:
      fpath: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
    '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=hp.sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag

def spectrogram2wav(mag, model):
    '''# Generate wave file from linear magnitude spectrogram

    Args:
      mag: A numpy array of (T, 1+n_fft//2)

    Returns:
      wav: A 1-D numpy array.
    '''
	
	# phase approximation
    phs = np.random.uniform(-math.pi,math.pi,mag.shape)
    #phs = np.zeros(mag.shape)
    if hp.phase_reconstruction == True:
        mag_temp = np.copy(mag)
        mag_to_phs = mag_temp[:,:hp.freq_bins_recon]
        mag_to_phs-=np.mean(mag_to_phs)
        mag_to_phs/=np.std(mag_to_phs)
        phs_approx = model.predict(mag_to_phs)
        phs[:,:hp.freq_bins_recon]=phs_approx[:,:]

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

	# transpose
    mag = mag.T
    phs = phs.T

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag**hp.power,phs)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

def griffin_lim(spectrogram, initial_phase):
    '''Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)
    X_best = X_best * np.exp(1j * initial_phase)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y
	
def fast_griffin_lim(spectrogram, initial_phase):
    X_t = None
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter_fast):
        if X_t is None:
            phase = initial_phase
        else:
            X_previous = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
            phase = np.angle(X_previous)

        X_previous = X_best * np.exp(1j * phase)
        X_best = spectrogram + (0.1 * np.abs(X_previous))
        X_t = invert_spectrogram(X_previous)

    y = np.real(X_t)
    return y

def invert_spectrogram(spectrogram):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")

def plot_alignment(alignment, gs, dir=hp.logdir):
    """Plots the alignment.

    Args:
      alignment: A numpy array with shape of (encoder_steps, decoder_steps)
      gs: (int) global step.
      dir: Output path.
    """
    if not os.path.exists(dir): os.mkdir(dir)

    fig, ax = plt.subplots()
    im = ax.imshow(alignment)

    fig.colorbar(im)
    plt.title('{} Steps'.format(gs))
    plt.savefig('{}/alignment_{}.png'.format(dir, gs), format='png')
    plt.close(fig)

def guided_attention(g=0.2):
    '''Guided attention. Refer to page 3 on the paper.'''
    W = np.zeros((hp.max_N, hp.max_T), dtype=np.float32)
    for n_pos in range(W.shape[0]):
        for t_pos in range(W.shape[1]):
            W[n_pos, t_pos] = 1 - np.exp(-(t_pos / float(hp.max_T) - n_pos / float(hp.max_N)) ** 2 / (2 * g * g))
    return W

def learning_rate_decay(init_lr, global_step, warmup_steps = 4000.0):
    '''Noam scheme from tensor2tensor'''
    step = tf.to_float(global_step + 1)
    return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)

def load_spectrograms(fpath):
    '''Read the wave file in `fpath`
    and extracts spectrograms'''

    fname = os.path.basename(fpath)
    mel, mag = get_spectrograms(fpath)
    t = mel.shape[0]

    # Marginal padding for reduction shape sync.
    num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")

    # Reduction
    mel = mel[::hp.r, :]
    return fname, mel, mag

