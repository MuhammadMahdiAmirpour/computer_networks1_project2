import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Read the audio file
sampling_rate, data = wavfile.read('resources/voice1.wav')

# Function to change the sampling rate
def change_sampling_rate(data, original_rate, speed_factor):
    """
    Change the sampling rate of the audio to speed up the playback.
    
    Parameters:
    data (np.array): The input audio signal.
    original_rate (int): The original sampling rate.
    speed_factor (float): The factor by which to speed up the audio.
    
    Returns:
    tuple: (new_data, new_sampling_rate)
    """
    # New sampling rate
    new_sampling_rate = int(original_rate * speed_factor)
    return data, new_sampling_rate

# Speed factor (e.g., 2 means twice as fast)
speed_factor = 2

# Change the sampling rate to increase speed
new_data, new_sampling_rate = change_sampling_rate(data, sampling_rate, speed_factor)

# Save the sped-up audio
output_filename = f'src/level6/voice1_sped_up_{speed_factor}x.wav'
wavfile.write(output_filename, new_sampling_rate, new_data)

# Plot original and sped-up signals for comparison
time_original = np.arange(data.shape[0]) / sampling_rate
time_sped_up = np.arange(new_data.shape[0]) / new_sampling_rate

plt.figure(figsize=(15, 6))

# Plot original signal
plt.subplot(2, 1, 1)
plt.plot(time_original, data[:, 0], label='Original Signal (Left Channel)')
plt.title('Original Signal')
plt.legend()

# Plot sped-up signal
plt.subplot(2, 1, 2)
plt.plot(time_sped_up, new_data[:, 0], label=f'Sped-Up Signal ({speed_factor}x, Left Channel)', color='orange')
plt.title(f'Sped-Up Signal ({speed_factor}x)')
plt.legend()

plt.tight_layout()
plt.show()

print(f'Original Sampling Rate: {sampling_rate} Hz')
print(f'New Sampling Rate: {new_sampling_rate} Hz')
print(f'Sped-up audio saved as: {output_filename}')
