# import pyaudio
import numpy as np
from scipy.io import wavfile
import soundfile as sf
import soundcard as sc
from scipy.signal import butter, lfilter
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib as mp
import os
# import pyaudio
import wave
import sys
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import hilbert
import simpleaudio as sa
from modulation_functions import *

###-----------###
### CONSTANTS ###
###-----------###
global FREQ
global FM_PM
global DURATION_PM
global SAMPLE_RATE

MAX_VOLUME = 32767
PI = np.pi
SAMPLE_RATE = 44000  # 1/Sec
BASIC_TIME = 5000 / SAMPLE_RATE  # ~0.1133 Sec
FREQ = 18500  # Hz
FREQ_SHIFT = 1000
samples = np.array(0.).astype(np.float32)
CHUNK_SIZE = 50
RECORD_TIME = 4

### PM constants ###
FM_PM = 400  # frequency of modulating signal
DURATION_PM = 2 / FM_PM  # 0.01#duration of the signal
ALPHA = PI / 2  # 0.3 #amplitude of modulating signal
THETA = PI / 4  # phase offset of modulating signal
BETA = PI / 5  # constant carrier phase offset

###------------------###
###  create gausian  ###
###------------------###

a = 1
b1 = 0
b2 = 1
c = 10 / SAMPLE_RATE


###------------------###

def get_gausian(basic_time=BASIC_TIME):
    samples_per_bit = int(basic_time * SAMPLE_RATE)

    arr1 = np.linspace(0, 0.5, samples_per_bit // 2)
    arr2 = np.linspace(0.5, 1, samples_per_bit // 2)

    gaus1 = lambda x: a * np.exp(-((x - b1) ** 2) / c)
    gaus2 = lambda x: a * np.exp(-((x - b2) ** 2) / c)

    res_arr1 = 1 - gaus1(arr1)
    res_arr2 = 1 - gaus2(arr2)
    arr = np.concatenate((arr1, arr2))
    res = np.concatenate((res_arr1, res_arr2))
    return res


def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a


def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


###--------------------###
### save/load wav file ###
###--------------------###
def save_wave(frame_rate, audio_data, wave_filename):
    try:
        data = np.asarray(audio_data)
        mask = np.mod(data, 1)
        if sum(mask == 0)[0] < data.shape[0] or sum(mask == 0)[1] < data.shape[0]:
            raise Exception('Invalid audio data')
        wavfile.write(wave_filename, frame_rate, data.astype(np.int16))
        return 0
    except KeyboardInterrupt:
        raise
    except:
        return -1


###---------------------------------###
### create the modulated sound wave ###
###---------------------------------###


def check_hilbert(m_t, x, t):
    ###
    plt.figure()
    plt.plot(t, m_t)  # plot modulating signal
    plt.title('Modulating signal')
    plt.xlabel('t')
    plt.ylabel('m(t)')

    nMean = 0  # noise mean
    nSigma = 0.1  # noise sigma
    n = np.random.normal(nMean, nSigma, len(t))
    r = x + n + np.sin(2 * PI * 400 * t)  # noisy received signal
    #     r = x

    r = butter_highpass_filter(r, FREQ, SAMPLE_RATE, order=8)

    # Demodulation of the noisy Phase Modulated signal
    z = hilbert(r)  # form the analytical signal from the received vector
    inst_phase = np.unwrap(np.angle(z))  # instaneous phase
    coef_ = np.polyfit(t, inst_phase, 1)
    offsetTerm = coef_[0] * t + coef_[1]
    demodulated = inst_phase - offsetTerm

    ###
    plt.figure()
    plt.plot(t, demodulated)  # demodulated signal
    plt.title('Demodulated signal')
    plt.xlabel('n')
    plt.ylabel('\hat{m(t)}')


###------------###
### MODULATION ###
###------------###


###--------------------###
### play from raw data ###
###--------------------###

def play_data(final_sample_list):
    samples = np.asarray(final_sample_list) / MAX_VOLUME
    default_speaker = sc.default_speaker()
    default_speaker.play(samples, samplerate=SAMPLE_RATE)


###--------###
### RECORD ###
###--------###

def record(file_name, record_time=RECORD_TIME):
    fs = SAMPLE_RATE  # this is the frequency sampling; also: 4999, 64000
    seconds = record_time  # Duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    print("Starting: Speak now!")
    print(fs)
    sd.wait()  # Wait until recording is finished
    print("finished")
    write(file_name, fs, myrecording)  # Save as WAV file
    # os.startfile("hello.wav")


# In[94]:

