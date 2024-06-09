import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import matplotlib.pyplot as plt

def multiply_volume(data, n):
    """
    Doubles the volume of the audio data by the specified factor.
    
    Args:
        data (numpy.ndarray): The audio data.
        factor (int, optional): The factor by which to double the volume. Default is 1.
    
    Returns:
        numpy.ndarray: The audio data with doubled volume.
    """
    return data * n

def save_wav(filename, data, samplerate):
    wav.write(filename, samplerate, data)

def play_audio(data, samplerate):
    sd.play(data, samplerate)
    sd.wait()

def plot_data(input_data):
    num = len(input_data)
    _, ax = plt.subplots(2 * num, 1, figsize=(8, 6))
    x = np.linspace(1, 100, data.shape[0])
    for i in range(0, 2 * num, 2):
        ax[i].plot(x, input_data[i//2][:, 0], color="red")
        ax[i+1].plot(x, input_data[i//2][:, 1], color="blue")
    plt.show()

def main(data, sampling_rate):
    data_list = list()
    # multiply the volume once
    for i in np.arange(-2, 4.1, 1).tolist():
        if i != 0:
            multiplied_data = multiply_volume(data, i)
            data_list.append(multiplied_data)
            save_wav(f"volume_multiplied_by_{i}.wav", multiplied_data, sampling_rate)
            play_audio(multiplied_data, sampling_rate)
            print(f"Playing audio with volume multiplied by {i}")
    plot_data(data_list)

if __name__ == "__main__":
    # Read the audio file
    sampling_rate, data = wav.read('../../resources/voice1.wav')
    
    # Call the main function with the audio data and sampling rate
    main(data, sampling_rate)

