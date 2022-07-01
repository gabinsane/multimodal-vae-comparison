try:
    import sounddevice as sd
    import soundfile as sf
except:
    print("Could not load audio libraries, training with audio might not work")
import numpy as np
import time, pickle, random

def read_mp3(path, normalized=False):
    """MP3 to numpy array"""
    import pydub
    a = pydub.AudioSegment.from_mp3(path)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

def write(f, sr, x, normalized=False):
    """numpy array to MP3"""
    import pydub
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")


def numpy_to_wav(filename, data, sampling_freq=24000):
    sf.write(filename, data, sampling_freq)

def add_noise(sound):
    add_silence = [random.randint(1, 1000), random.randint(1, 200)]
    noise = np.random.normal(1,random.randint(1,1000)/1000,sound.shape[0])
    sound_noise = sound.copy()
    for x in range(sound.shape[1]):
        sound_noise[:, x] = sound[:, 1] * noise
    sound_padded = sound_noise.copy()
    for x in range(add_silence[0]):
        sound_padded = np.insert(sound_padded, 0, np.array((0, 0)), 0)
    for x in range(add_silence[1]):
        sound_padded = np.insert(sound_padded, -1,np.array((0, 0)), 0)
    return sound_padded

def play_as_sound(signal, sampling_rate=24000):
    """
    Plays the input signal as audio
    :param signal: np.array, signal
    :param sampling_rate: int, 24000 for original files, 12000 for compressed
    """
    sd.play(signal, sampling_rate)


def inspect_dataset(dataset_path):
    with open(dataset_path, 'rb') as h:
        data = pickle.load(h)
    for sig in data:
        play_as_sound(sig.astype(np.int16), 16000)
        time.sleep(3)
