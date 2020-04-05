from smop import *
from smop_functions import *
from modulation_functions import *


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
    NewForTrans = ForTrans
    # for index in indexes:
    #     if np.max(ForTrans[index]) > thresh:
    #         NewForTrans.append(ForTrans[index])
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
