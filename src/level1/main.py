import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import matplotlib.pyplot as plt

def double_volume(data):
    """
    Doubles the volume of the audio data by the specified factor.
    
    Args:
        data (numpy.ndarray): The audio data.
        factor (int, optional): The factor by which to double the volume. Default is 1.
    
    Returns:
        numpy.ndarray: The audio data with doubled volume.
    """
    return data * 2

def save_wav(filename, data, samplerate):
    wav.write(filename, samplerate, data)

def play_audio(data, samplerate):
    sd.play(data, samplerate)
    sd.wait()

def plot_data(input_data):
    x = np.linspace(1, 100, data.shape[0])
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))
    ax1.plot(x, input_data[0])
    ax2.plot(x, input_data[1])
    ax3.plot(x, input_data[2])
    plt.show()

def main(data, sampling_rate):
    # Double the volume once
    doubled_data = double_volume(data)
    save_wav('doubled_volume.wav', doubled_data, sampling_rate)
    play_audio(doubled_data, sampling_rate)
    print('Playing audio with volume doubled once')
    
    # Double the volume 4 times (4x)
    quadrupled_data = double_volume(double_volume(data))
    save_wav('quadrupled_volume.wav', quadrupled_data, sampling_rate)
    play_audio(quadrupled_data, sampling_rate)
    print('Playing audio with volume quadrupled (4x)')
    plot_data((data, doubled_data, quadrupled_data))

if __name__ == "__main__":
    # Read the audio file
    sampling_rate, data = wav.read('../../resources/voice1.wav')
    
    # Call the main function with the audio data and sampling rate
    main(data, sampling_rate)

