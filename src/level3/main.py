import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import matplotlib.pyplot as plt

mu = 255
A = 87.6

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

def mu_law_encode(signal, mu):
    """
    Perform μ-law encoding (non-linear quantization).
    
    Parameters:
    signal (numpy array): Input signal to be encoded.
    mu (int): μ parameter for μ-law companding.
    
    Returns:
    numpy array: μ-law encoded signal.
    """
    # Normalize the input signal to be in the range [-1, 1]
    signal = signal / np.max(np.abs(signal))
    
    # Perform μ-law encoding
    encoded_signal = np.sign(signal) * np.log1p(mu * np.abs(signal)) / np.log1p(mu)
    return encoded_signal

def mu_law_decode(encoded_signal, mu):
    """
    Perform μ-law decoding.
    
    Parameters:
    encoded_signal (numpy array): μ-law encoded signal to be decoded.
    mu (int): μ parameter for μ-law companding.
    
    Returns:
    numpy array: Decoded signal.
    """
    # Perform μ-law decoding
    decoded_signal = np.sign(encoded_signal) * (1 / mu) * ((1 + mu) ** np.abs(encoded_signal) - 1)
    return decoded_signal

def a_law_encode(signal, A, epsilon=1e-12):
    """
    Perform A-law encoding (non-linear quantization).
    
    Parameters:
    signal (numpy array): Input signal to be encoded.
    A (float): A parameter for A-law companding.
    epsilon (float): Small value to avoid log of zero.
    
    Returns:
    numpy array: A-law encoded signal.
    """
    # Normalize the input signal to be in the range [-1, 1]
    signal = signal / np.max(np.abs(signal))
    
    # Prevent zero values by adding a small epsilon
    abs_signal = np.abs(signal) + epsilon
    
    # Perform A-law encoding
    encoded_signal = np.where(
        abs_signal < 1 / A,
        A * abs_signal / (1 + np.log(A)),
        (1 + np.log(A * abs_signal)) / (1 + np.log(A))
    )
    return np.sign(signal) * encoded_signal

def a_law_decode(encoded_signal, A):
    """
    Perform A-law decoding.
    
    Parameters:
    encoded_signal (numpy array): A-law encoded signal to be decoded.
    A (float): A parameter for A-law companding.
    
    Returns:
    numpy array: Decoded signal.
    """
    # Perform A-law decoding
    abs_encoded_signal = np.abs(encoded_signal)
    decoded_signal = np.where(
        abs_encoded_signal < 1 / (1 + np.log(A)),
        abs_encoded_signal * (1 + np.log(A)) / A,
        np.exp(abs_encoded_signal * (1 + np.log(A)) - 1) / A
    )
    return np.sign(encoded_signal) * decoded_signal

def main(data, sampling_rate):
    data_list = list()
    mu_encoded_list = list()
    mu_decoded_list = list()
    a_encoded_list = list()
    a_decoded_list = list()
    # multiply the volume once
    for i in np.arange(-2, 4.1, 1).tolist():
        if i != 0:
            multiplied_data = multiply_volume(data, i)
            data_list.append(multiplied_data)
            mu_encoded = mu_law_encode(multiplied_data, mu)
            mu_decoded = mu_law_decode(mu_encoded, mu)
            mu_encoded_list.append(mu_encoded)
            mu_decoded_list.append(mu_decoded)
            a_encoded = a_law_encode(multiplied_data, A)
            a_decoded = a_law_decode(a_encoded, A)
            a_encoded_list.append(a_encoded)
            a_decoded_list.append(a_decoded)
            save_wav(f"volume_multiplied_by_{i}.wav", multiplied_data, sampling_rate)
            play_audio(multiplied_data, sampling_rate)
            print(f"Playing audio with volume multiplied by {i}")
    plot_data(data_list)
    plot_data(mu_encoded_list)
    plot_data(mu_decoded_list)
    plot_data(a_encoded_list)
    plot_data(a_decoded_list)

    for i, encoded in enumerate(mu_encoded_list):
        save_wav(f"mu_encoded_volume_multiplied_by_{-2 + i*1}.wav", encoded, sampling_rate)
    for i, decoded in enumerate(mu_decoded_list):
        save_wav(f"mu_decoded_volume_multiplied_by_{-2 + i*1}.wav", decoded, sampling_rate)
    for i, encoded in enumerate(a_encoded_list):
        save_wav(f"a_encoded_volume_multiplied_by_{-2 + i*1}.wav", encoded, sampling_rate)
    for i, decoded in enumerate(a_decoded_list):
        save_wav(f"a_decoded_volume_multiplied_by_{-2 + i*1}.wav", decoded, sampling_rate)


if __name__ == "__main__":
    # Read the audio file
    sampling_rate, data = wav.read('../../resources/voice1.wav')
    
    if data.ndim == 2:
        data = data.mean(axis=1)
    
    data = data / np.max(np.abs(data))
    
    # Call the main function with the audio data and sampling rate
    print(data.shape)
    main(data, sampling_rate)
