import pyttsx3
import sounddevice as sd
import soundfile as sf
import pydub
import numpy as np
import os, time, pickle, random
from itertools import chain
noise_types = {"f":{1:random.randint(90,300), 2:False},"fs":{1:random.randint(100,130), 2:False}, "t":{1:220, 2:True}, "s":{1:125, 2:False}, "0":{1:125, 2:False}, "ts":{1:220, 2:True},}

path1 = "./jsonFiles/sounddata/{}_soundset.pkl"

# def mp3_to_numpy(path):
#     signal, sampling_rate = open_audio(path)
#     return signal

def read_mp3(path, normalized=False):
    """MP3 to numpy array"""
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
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")


def numpy_to_wav(filename, data, sampling_freq=24000):
    #file = sd.rec(data, sampling_freq, channels=2, blocking=True)
    sf.write(filename, data, sampling_freq)

def get_noisy_sound_sample(category):
    sound = get_soundset_sample(category)
    sound_noisy = add_noise(sound)
    return np.asarray(sound_noisy)

def add_noise(sound):
    add_silence = [random.randint(1, 1000), random.randint(1, 200)]
    noise = np.random.normal(1,random.randint(1,1000)/1000,sound.shape[0])
    sound_noise = sound.copy()
    for x in range(sound.shape[1]):
        sound_noise[:, x] = sound[:, 1] * noise
    sound_padded = sound_noise.copy()
    #print(np.nonzero(sound_padded!=0)[0][0])
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
        print("Playing sound")
        play_as_sound(sig.astype(np.int16), 16000)
        time.sleep(3)

def make_soundsets(actions, path):
    allsounds = next(os.walk(path), (None, None, []))[2]
    soundsets = [[],[],[]]
    for snd in allsounds:
        ph = os.path.join(path1, snd)
        fr, sound_arr = read_mp3(ph)
        if len(sound_arr.shape) < 2:
            print(snd)
        for ix, x in enumerate(actions):
            if x in snd:
                soundsets[ix].append(sound_arr)
    print("Data merged")
    for ix, soundset in enumerate(soundsets):
        with open(os.path.join(path, "{}_soundset.pkl".format(actions[ix])), 'wb') as handle:
            pickle.dump(soundset, handle)

def get_soundset_sample(cat):
    with open(path1.format(cat.lower()), 'rb') as h:
        data = list(pickle.load(h))
    return random.choice(data)


if __name__ == "__main__":
    inspect_dataset("/home/gabi/mirracle_remote/mirracle_multimodal/mirracle_multimodal/results/exp_sounds5/visuals/reconstructions.pkl")
    #actions = ["lift", "bump", "push"]
    #make_soundsets(actions, path1)

