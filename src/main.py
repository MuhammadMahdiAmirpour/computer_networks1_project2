import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import sounddevice as sd

sampling_rate, data = wav.read('voice1.wav')
print('sampling rate:', sampling_rate)
print('data type:', data.dtype)
print('data shape:', data.shape)
N, no_channels = data.shape
print('signal length:', N)
channel0 = data[:, 0]
channel1 = data[:, 1]

def save_wav(filename, data, samplerate):
    wav.write(filename, samplerate, data)

def play_audio(data, samplerate):
    sd.play(data, samplerate)
    sd.wait()

def main():
    linear_data = linear_pcm(data)
    save_wav('linear_pcm.wav', linear_data, sampling_rate)
    play_audio(linear_data, sampling_rate)
    print('Playing Linear PCM')

if __name__ == "__main__":
    main()