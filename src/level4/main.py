import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import matplotlib.pyplot as plt

step_size = 0.1

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
    _, ax = plt.subplots(num, 1, figsize=(8, 6))
    x = np.linspace(1, 100, data.shape[0])
    for i in range(num):
        ax[i].plot(x, input_data[i])
    plt.show()

def plot_data(input_data):
    num = len(input_data)
    _, ax = plt.subplots(num, 1, figsize=(8, 6))
    x = np.linspace(0, len(input_data[0]), 500)
    for i in range(num):
        ax[i].plot(x, input_data[i][:500])
    plt.show()



def main(data, sampling_rate):
    data_list = list()
    bitstream_list = list()
    decoded_audio_list = list()
    
    for i in np.arange(-2, 4.1, 1).tolist():
        if i != 0:
            multiplied_data = multiply_volume(data, i)
            data_list.append(multiplied_data)
            
            # Perform delta modulation
            decoded_audio, bitstream, sampling_rate = delta_modulation(sampling_rate, data)
            decoded_audio_list.append(decoded_audio)
            bitstream_list.append(bitstream_list)
            
            play_audio(multiplied_data, sampling_rate)
            print(f"Playing audio with volume multiplied by {i}")

    # Optionally plot the delta-modulated signals
    plot_data(decoded_audio_list)
    plot_data(bitstream_list)

    # Optionally save the delta-modulated signals to .wav files
    for i, dm_signal in enumerate(decoded_audio_list):
        save_wav(f"dm_signal_volume_multiplied_by_{-2 + i*1}.wav", dm_signal, sampling_rate)

if __name__ == "__main__":
    # Read the audio file
    sampling_rate, data = wav.read('../../resources/voice1.wav')
    
    # Call the main function with the audio data and sampling rate
    main(data, sampling_rate)
