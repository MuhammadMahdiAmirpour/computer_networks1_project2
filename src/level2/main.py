import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import matplotlib.pyplot as plt

def multiply_data_by_range(data, start, end):
    x = np.linspace(start, end, len(data), dtype=np.float64)
    x[x == 0] = 1e-10
    data[:, 0] *= x
    data[:, 1] *= x

def save_wav(filename, data, samplerate):
    wav.write(filename, samplerate, data)

def play_audio(data, samplerate):
    sd.play(data, samplerate)
    sd.wait()

def plot_data(input_data):
    _, ax = plt.subplots(2, 1, figsize=(8, 6))
    x = np.linspace(1, 100, data.shape[0])
    ax[0].plot(x, input_data[:, 0], color="red")
    ax[1].plot(x, input_data[:, 1], color="blue")
    plt.show()

def main(data, sampling_rate):
    # multiply the volume once
    multiply_data_by_range(data, -2, 4)
    save_wav(f"src/level2/volume_multiplied.wav", data, sampling_rate)
    play_audio(data, sampling_rate)
    plot_data(data)

def normalize_channel(channel_data):
    max_val = np.max(channel_data)
    channel_data /= max_val
    return channel_data

if __name__ == "__main__":
    # Read the audio file
    sampling_rate, data = wav.read('resources/voice1.wav')
    data = data.astype(np.float64)
    data[:, 0] = normalize_channel(data[:, 0])
    data[:, 1] = normalize_channel(data[:, 1])
    
    # Call the main function with the audio data and sampling rate
    main(data, sampling_rate)

