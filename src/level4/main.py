import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def delta_modulation(signal, step_size):
    """
    Perform delta modulation on a given signal.
    
    Parameters:
    signal (np.array): The input audio signal.
    step_size (float): The fixed step size for the modulation.
    
    Returns:
    np.array: The delta modulated binary signal.
    np.array: The reconstructed signal from delta modulation.
    """
    # Ensure the signal is one-dimensional (convert stereo to mono if necessary)
    if signal.ndim == 2:
        num_channels = signal.shape[1]
        binary_output = np.zeros_like(signal)
        reconstructed_signal = np.zeros_like(signal)
        
        for channel in range(num_channels):
            # Initialize the variables
            N = signal.shape[0]
            reconstructed_signal[0, channel] = signal[0, channel]
            
            # Delta modulation process
            for i in range(1, N):
                if signal[i, channel] > reconstructed_signal[i - 1, channel]:
                    binary_output[i, channel] = 1
                    reconstructed_signal[i, channel] = reconstructed_signal[i - 1, channel] + step_size
                else:
                    binary_output[i, channel] = 0
                    reconstructed_signal[i, channel] = reconstructed_signal[i - 1, channel] - step_size
    else:
        N = len(signal)
        binary_output = np.zeros(N)
        reconstructed_signal = np.zeros(N)
        reconstructed_signal[0] = signal[0]
        
        # Delta modulation process
        for i in range(1, N):
            if signal[i] > reconstructed_signal[i - 1]:
                binary_output[i] = 1
                reconstructed_signal[i] = reconstructed_signal[i - 1] + step_size
            else:
                binary_output[i] = 0
                reconstructed_signal[i] = reconstructed_signal[i - 1] - step_size
    
    return binary_output, reconstructed_signal

# Read the audio file
sampling_rate, data = wavfile.read('resources/voice1.wav')

# Normalize the data
data = data / np.max(np.abs(data))

# Set the step size
step_size = 0.05

# Perform delta modulation
binary_output, reconstructed_signal = delta_modulation(data, step_size)
wavfile.write("src/level4/dm_output.wav", sampling_rate, reconstructed_signal)

# Plot the original and reconstructed signals
plt.figure(figsize=(15, 12))

# Original signal (left channel)
plt.subplot(4, 1, 1)
plt.plot(data[:, 0], label='Original Signal (Left Channel)')
plt.title('Original Signal (Left Channel)')
plt.legend()

# Reconstructed signal (left channel)
plt.subplot(4, 1, 2)
plt.plot(reconstructed_signal[:, 0], label='Reconstructed Signal (Left Channel)', color='orange')
plt.title('Reconstructed Signal from Delta Modulation (Left Channel)')
plt.legend()

# Original signal (right channel)
plt.subplot(4, 1, 3)
plt.plot(data[:, 1], label='Original Signal (Right Channel)')
plt.title('Original Signal (Right Channel)')
plt.legend()

# Reconstructed signal (right channel)
plt.subplot(4, 1, 4)
plt.plot(reconstructed_signal[:, 1], label='Reconstructed Signal (Right Channel)', color='orange')
plt.title('Reconstructed Signal from Delta Modulation (Right Channel)')
plt.legend()

plt.tight_layout()
plt.show()
