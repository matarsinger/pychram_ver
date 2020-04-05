from smop import *
from smop_functions import *


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
