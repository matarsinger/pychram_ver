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

###-----------###
### CONSTANTS ###
###-----------###
global FREQ
global FM_PM
global DURATION_PM
global SAMPLE_RATE

MAX_VOLUME = 32767
PI = np.pi
SAMPLE_RATE = 44100  # 1/Sec
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


def createBitArray(bits, n):
    l = len(bits)
    bit_array = []
    res = get_gausian()
    bit_len = int(BASIC_TIME * SAMPLE_RATE)
    for k in range(l):
        bit_array.append(np.ones(bit_len) * bits[k] * res)

    final_bit_array = bit_array[0]
    for arr in range(1, len(bit_array)):
        final_bit_array = np.append(final_bit_array, bit_array[arr])

    return final_bit_array


def write_raw_samples(samples, len_bits, delta, theta):
    samples = SAMPLE_RATE / int(FREQ + delta)
    num_of_samples = int(SAMPLE_RATE * len_bits * BASIC_TIME)
    samples_list = np.arange(num_of_samples)

    sample_list = MAX_VOLUME * np.cos(2 * PI * samples_list / samples + theta)

    return sample_list, num_of_samples


def write_notes(bits, len_bits, theta=0, delta=0):
    sample_list, num_of_samples = write_raw_samples(samples, len_bits, delta, theta)

    bit_array = createBitArray(bits, num_of_samples)
    sample_list = sample_list * bit_array

    return sample_list.astype(int)


def write_sample_list(bits, d=0, t=0, n=1):
    final_sample_list = []

    ### create the one fitted ###
    one_array = write_notes(bits, len(bits), delta=d * FREQ_SHIFT)

    ### create the zero fitted ###
    zeros_array = write_notes(1 - bits, len(bits), theta=t * PI)
    final_array = one_array + zeros_array * n
    for samp in final_array:
        final_sample_list.append([samp, samp])

    return final_sample_list


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


### PM bit array ###
def createBitArrayPM(bits):
    """
    creates the array of bits to multiply the sin wave
    """
    l = len(bits)
    bit_array = []
    res = get_gausian(DURATION_PM)
    bit_len = int(DURATION_PM * SAMPLE_RATE)
    for k in range(l):
        bit_array.append(np.ones(bit_len) * bits[k])
    final_bit_array = bit_array[0]
    for arr in range(1, len(bit_array)):
        final_bit_array = np.append(final_bit_array, bit_array[arr])
    return final_bit_array


def PM_modulation(bits):
    t = np.arange(int(SAMPLE_RATE * DURATION_PM) * len(bits)) / SAMPLE_RATE  # time base

    info = np.pi * createBitArrayPM(bits)
    print('info = ', len(info))
    print('t = ', len(t))

    # Phase Modulation
    m_t = info * np.cos(2 * PI * FM_PM * t + info)
    x = np.cos(2 * PI * FREQ * t + BETA + m_t)  # modulated signal
    check_hilbert(m_t, x, t)
    final_sample_list = []

    x = x * MAX_VOLUME
    x = x.astype(int)
    for samp in x:
        final_sample_list.append([samp, samp])

    dat = redecyherPM(DecypherPM(x, thresh=1))
    print(dat)

    return final_sample_list


###------------###
### MODULATION ###
###------------###

def modulation(to_transmit, mode=None):
    if mode == "freq":
        ### simple FREQUENCY_SHIFT modulation
        final_sample_list = write_sample_list(to_transmit, d=1)

    elif mode == "binary":
        ### simple BINARY_SHIFT modulation
        final_sample_list = write_sample_list(to_transmit, n=0)

    elif mode == "phase":
        ### simple PHASE_SHIFT modulation
        final_sample_list = write_sample_list(to_transmit, t=1)
    elif mode == "PM":
        ### PM with carrier wave
        final_sample_list = PM_modulation(to_transmit)

    else:
        print("give me something")
        return None

    return final_sample_list


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


def DecypherPM(r, thresh):
    chunk_size = 200
    t = np.arange(len(r))
    r = butter_highpass_filter(r, FREQ, SAMPLE_RATE, order=8)
    z = hilbert(r)  # form the analytical signal from the received vector
    inst_phase = np.unwrap(np.angle(z))  # instaneous phase
    chunk_inst_phase = [inst_phase[i * chunk_size: (i + 1) * chunk_size] for i in range(len(inst_phase) // chunk_size)]
    t_chunk = [t[i * chunk_size:(i + 1) * chunk_size] for i in range(len(t) // chunk_size)]

    for i in range(len(chunk_inst_phase)):
        coef_ = np.polyfit(t_chunk[i], chunk_inst_phase[i], 1)
        offset = coef_[0] * t_chunk[i] + coef_[1]
        dem = np.abs(chunk_inst_phase[i] - offset)
        dem_chunk = None
        if i == 0:
            dem_chunk = dem
        else:
            dem_chunk = np.append(dem_chunk, [dem])

    dem_chunk[dem_chunk > thresh] = thresh
    print(np.max(dem_chunk))

    t = np.arange(len(dem_chunk))
    plt.figure()
    plt.plot(t, np.abs(dem_chunk))

    #     plt.xlim(0, 1000)
    #     plt.ylim(5, 20)
    return dem_chunk


def deal_queue(item, max_size, queue):
    queue.insert(0, item)
    if len(queue) > max_size:
        queue.pop()
    return queue


def redecyherPM(data, ChunkSize=int(DURATION_PM * SAMPLE_RATE), min_thresh=0.5, queue_max_size=10, sup=1):
    data = abs(data)
    thresh = min_thresh
    queue = []
    NumOfChunks = len(data) / ChunkSize
    ChunkList = np.array_split(data, NumOfChunks)
    data_num_of_times = []
    for chunk in ChunkList:
        if np.sum(chunk) / ChunkSize > thresh:
            data_num_of_times.append(1)
            if np.sum(chunk) / ChunkSize < sup:
                queue = deal_queue(np.sum(chunk) / ChunkSize, queue_max_size, queue)
        else:
            data_num_of_times.append(0)
        if len(queue) > 5:
            thresh = 6 / 10 * np.sum(queue) / len(queue)
    #     plt.figure()
    #     plt.plot(data_num_of_times)
    return data_num_of_times


###---------------------###
### cut irelevant parts ###
###---------------------###

def cut_irellevant(data, chunk_size=100, thresh=0.2):
    l = len(data) // chunk_size

    ### Do FFT by chunks
    chunk_array = [np.array(data[chunk_size * i:(i + 1) * chunk_size]) for i in range(l)]
    freq = np.fft.fftfreq(chunk_size, d=1 / SAMPLE_RATE)
    fft_array = [np.abs(np.fft.fft(arr)) for arr in chunk_array]

    ### Integral over relevant freqs
    cut_index = int(3 * chunk_size / 8)
    integral_array = [np.trapz(arr[cut_index:chunk_size // 2], freq[cut_index:chunk_size // 2]) for arr in fft_array]
    integral_array = np.asarray(integral_array) / (-freq[chunk_size // 2] + freq[cut_index])

    ### Filter by thresh
    good_chunks = [i for i in range(len(integral_array)) if integral_array[i] > thresh]
    plt.figure()
    plt.plot(integral_array)

    begin = good_chunks[0] * chunk_size
    end = (good_chunks[-1] + 1) * chunk_size

    ### Cut and return data
    return data[begin:end]


# In[94]:


###--------###
### DECODE ###
###--------###

def DecypherFreqShift(data, ChunkSize=CHUNK_SIZE, thresh=0.4):
    #     sos = signal.butter(4,FREQ - FREQ_SHIFT, btype = 'hp', fs=SAMPLE_RATE, output='sos')
    #     filtered = signal.sosfilt(sos, data)
    #     data = np.array(filtered[:,0])
    #     data = np.array(data[:, 0])

    NumOfChunks = len(data) / ChunkSize
    ChunkList = np.array_split(data, NumOfChunks)
    ForTrans = [np.abs(np.fft.fft(dat)) for dat in ChunkList]
    indexes = range(len(ForTrans))
    ForTrans = [[0] * (ChunkSize // 2 - 1) +
                list(ForTrans[index][ChunkSize // 2:3 * ChunkSize // 4]) +
                [0] * (ChunkSize // 4 + 1) for index in indexes]
    NewForTrans = []
    for index in indexes:
        if np.max(ForTrans[index]) > thresh:
            NewForTrans.append(ForTrans[index])
    freqs = {index: np.fft.fftfreq(ChunkSize, d=1 / SAMPLE_RATE) for index in range(len(NewForTrans))}
    freqs_found = np.array([freqs[index]
                            [np.argmax(NewForTrans[index])]
                            for index in range(len(NewForTrans))])
    found = [1 if np.abs(FREQ + FREQ_SHIFT - np.abs(frequ))
                  < np.abs(FREQ - np.abs(frequ)) else 0 for frequ in freqs_found]
    #     print(found)
    return found


def RestoreChunks(DecyphData, Ratio):
    ChunkData = []
    Chunk = 0
    first = DecyphData[0]
    for i in range(len(DecyphData)):
        if DecyphData[i] == first:
            Chunk += 1
        else:
            first = DecyphData[i]
            ChunkData.append(Chunk)
            Chunk = 1
    ChunkData.append(Chunk)
    Sizes = [chunk // Ratio if chunk % Ratio < Ratio / 2
             else chunk // Ratio + 1 for chunk in ChunkData]
    data = []
    state = DecyphData[0]
    for i in Sizes:
        data = data + [state] * i
        state = 1 - state
    return data
