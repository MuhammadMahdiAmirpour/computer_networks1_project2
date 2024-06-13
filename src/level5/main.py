import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def quantize(signal, bits):
    """
    Quantize the input signal to the specified number of bits.
    
    Parameters:
    signal (np.array): The input audio signal.
    bits (int): The number of bits to quantize to.
    
    Returns:
    np.array: The quantized signal.
    """
    # Calculate the number of quantization levels
    quant_levels = 2 ** bits
    
    # Normalize signal to range 0 to 1
    signal_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    
    # Scale signal to the range of quantizer
    signal_scaled = signal_norm * (quant_levels - 1)
    
    # Quantize signal
    signal_quantized = np.round(signal_scaled)
    
    # Scale back to original range
    signal_quantized = signal_quantized / (quant_levels - 1)
    signal_quantized = signal_quantized * (np.max(signal) - np.min(signal)) + np.min(signal)
    
    return signal_quantized

# Read the audio file
sampling_rate, data = wavfile.read('resources/voice1.wav')

# Normalize the data
data = data / np.max(np.abs(data), axis=0)

# Define bit depths for quantization
bit_depths = [2, 4, 8, 16]

# Quantize and save the audio data for different bit depths
for bits in bit_depths:
    quantized_data = np.zeros_like(data)
    
    # Quantize each channel independently
    for channel in range(data.shape[1]):
        quantized_data[:, channel] = quantize(data[:, channel], bits)
    
    # Save the quantized audio data
    output_filename = f'src/level5/voice1_quantized_{bits}bits.wav'
    wavfile.write(output_filename, sampling_rate, np.int16(quantized_data * 32767))
    
    # Plot the quantized signal for visualization
    plt.figure(figsize=(10, 8))
    
    # Plot original and quantized signals for each channel
    plt.subplot(2, 1, 1)
    plt.plot(data[:1000, 0], label='Original Signal (Left Channel)')
    plt.plot(quantized_data[:1000, 0], label=f'Quantized Signal ({bits} bits, Left Channel)')
    plt.title(f'Original vs Quantized Signal ({bits} bits, Left Channel)')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(data[:1000, 1], label='Original Signal (Right Channel)')
    plt.plot(quantized_data[:1000, 1], label=f'Quantized Signal ({bits} bits, Right Channel)')
    plt.title(f'Original vs Quantized Signal ({bits} bits, Right Channel)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


# additional steps for audio comparison

from pydub import AudioSegment
from pydub.playback import play
import IPython.display as ipd

# Function to play an audio file
def play_audio(file_path):
    audio = AudioSegment.from_wav(file_path)
    play(audio)
    return ipd.Audio(file_path)

# Play the original audio
print("Original Audio:")
ipd.display(ipd.Audio('resources/voice1.wav'))

# Play the quantized audios
for bits in bit_depths:
    output_filename = f'src/level5/voice1_quantized_{bits}bits.wav'
    print(f"Quantized Audio ({bits} bits):")
    ipd.display(play_audio(output_filename))


