# mu = 256
# A = 87.6

# # def multiply_data_by_range(data, start, end):
# #     data[:, 0] *= np.linspace(start, end, len(data), dtype=np.int16)
# #     data[:, 1] *= np.linspace(start, end, len(data), dtype=np.int16)

# def encode_mu_law(signal, mu=256):
#     """
#     Perform μ-law encoding (non-linear quantization).
    
#     Parameters:
#     signal (numpy array): Input signal to be encoded.
#     mu (int): μ parameter for μ-law companding.
    
#     Returns:
#     numpy array: μ-law encoded signal.
#     """
#     # Normalize the input signal to be in the range [-1, 1]

#     sig_cop = np.copy(signal)
#     sig_cop[:, 0] = sig_cop[:, 0] / np.max(np.abs(sig_cop[:, 0]))
#     sig_cop[:, 1] = sig_cop[:, 1] / np.max(np.abs(sig_cop[:, 1]))
    
#     # Perform μ-law encoding
#     encoded_signal = np.copy(signal)
#     encoded_signal[:, 0] = np.sign(signal[:, 0]) * np.log1p(mu * np.abs(signal[:, 0])) / np.log1p(mu)
#     encoded_signal[:, 1] = np.sign(signal[:, 1]) * np.log1p(mu * np.abs(signal[:, 1])) / np.log1p(mu)
#     return encoded_signal

# def decode_mu_law(encoded_signal, mu=256):
#     """
#     Perform μ-law decoding.
    
#     Parameters:
#     encoded_signal (numpy array): μ-law encoded signal to be decoded.
#     mu (int): μ parameter for μ-law companding.
    
#     Returns:
#     numpy array: Decoded signal.
#     """
#     # Perform μ-law decoding
#     decoded_signal = np.copy(encoded_signal)
#     decoded_signal[:, 0] = np.sign(encoded_signal[:, 0]) * (1 / mu) * ((1 + mu) ** np.abs(encoded_signal[:, 0]) - 1)
#     decoded_signal[:, 1] = np.sign(encoded_signal[:, 1]) * (1 / mu) * ((1 + mu) ** np.abs(encoded_signal[:, 1]) - 1)
#     return decoded_signal

# def encode_a_law(signal, A=87.6, epsilon=1e-12):
#     """
#     Perform A-law encoding (non-linear quantization).
    
#     Parameters:
#     signal (numpy array): Input signal to be encoded.
#     A (float): A parameter for A-law companding.
#     epsilon (float): Small value to avoid log of zero.
    
#     Returns:
#     numpy array: A-law encoded signal.
#     """
#     # Normalize the input signal to be in the range [-1, 1]
#     sig_cop = np.copy(signal)
#     sig_cop[:, 0] = sig_cop[:, 0] / np.max(np.abs(sig_cop[:, 0]))
#     sig_cop[:, 1] = sig_cop[:, 1] / np.max(np.abs(sig_cop[:, 1]))
    
#     # Prevent zero values by adding a small epsilon
#     abs_signal = np.sign(signal)
#     abs_signal[:, 0] = np.abs(signal[:, 0]) + epsilon
#     abs_signal[:, 1] = np.abs(signal[:, 1]) + epsilon
    
#     # Perform A-law encoding
#     encoded_signal = np.zeros_like(signal)
#     encoded_signal[:, 0] = np.where(
#         abs_signal[:, 0] < 1 / A,
#         A * abs_signal[:, 0] / (1 + np.log(A)),
#         (1 + np.log(A * abs_signal[:, 0])) / (1 + np.log(A))
#     )
#     encoded_signal[:, 1] = np.where(
#         abs_signal[:, 1] < 1 / A,
#         A * abs_signal[:, 1] / (1 + np.log(A)),
#         (1 + np.log(A * abs_signal[:, 1])) / (1 + np.log(A))
#     )
#     return np.sign(signal) * encoded_signal

# def decode_a_law(encoded_signal, A=87.6):
#     """
#     Perform A-law decoding.
    
#     Parameters:
#     encoded_signal (numpy array): A-law encoded signal to be decoded.
#     A (float): A parameter for A-law companding.
    
#     Returns:
#     numpy array: Decoded signal.
#     """
#     # Perform A-law decoding
#     abs_encoded_signal = np.zeros_like(encoded_signal)
#     abs_encoded_signal[:, 0] = np.abs(encoded_signal[:, 0])
#     abs_encoded_signal[:, 1] = np.abs(encoded_signal[:, 1])
#     decoded_signal = np.zeros_like(encoded_signal)
#     decoded_signal[:, 0] = np.where(
#         abs_encoded_signal[:, 0] < 1 / (1 + np.log(A)),
#         abs_encoded_signal[:, 0] * (1 + np.log(A)) / A,
#         np.exp(abs_encoded_signal[:, 0] * (1 + np.log(A)) - 1) / A
#     )
#     decoded_signal[:, 1] = np.where(
#         abs_encoded_signal[:, 1] < 1 / (1 + np.log(A)),
#         abs_encoded_signal[:, 1] * (1 + np.log(A)) / A,
#         np.exp(abs_encoded_signal[:, 1] * (1 + np.log(A)) - 1) / A
#     )
#     return np.sign(encoded_signal) * decoded_signal

# def save_wav(filename, data, samplerate):
#     wav.write(filename, samplerate, data)

# def plot_data(input_data):
#     _, ax = plt.subplots(2, 1, figsize=(8, 6))
#     x = np.linspace(1, 100, data.shape[0])
#     ax[0].plot(x, input_data[:, 0], color="red")
#     ax[1].plot(x, input_data[:, 1], color="blue")
#     plt.show()

# def play_audio(data, samplerate):
#     sd.play(data, samplerate)
#     sd.wait()

# def main(data, sampling_rate):
#     # multiply the volume once
#     multiply_data_by_range(data, -2, 4)
#     print("ploting data")
#     plot_data(data)
#     mu_encoded = mu_law_encode(data, mu)
#     mu_decoded = mu_law_decode(mu_encoded, mu)
#     a_encoded = a_law_encode(data, A)
#     a_decoded = a_law_decode(a_encoded, A)
#     save_wav("src/level3/volume_multiplied.wav", data, sampling_rate)
#     play_audio(data, sampling_rate)
#     print("Playing audio with volume multiplied")
#     print("ploting mu_encoded data")
#     plot_data(mu_encoded)
#     print("ploting mu_decoded data")
#     plot_data(mu_decoded)
#     print("ploting a_encoded data")
#     plot_data(a_encoded)
#     print("ploting a_decoded data")
#     plot_data(a_decoded)

#     save_wav("src/level3/mu_encoded_volume_multiplied.wav", mu_encoded, sampling_rate)
#     save_wav("src/level3/mu_decoded_volume_multiplied.wav", mu_decoded, sampling_rate)
#     save_wav("src/level3/a_encoded_volume_multiplied.wav", a_encoded, sampling_rate)
#     save_wav("src/level3/a_decoded_volume_multiplied.wav", a_decoded, sampling_rate)


# if __name__ == "__main__":
#     # Read the audio file
#     sampling_rate, data = wav.read('resources/voice1.wav')
    
#     # if data.ndim == 2:
#         # data = data.mean(axis=1)
#     # data = data / np.max(np.abs(data))
    
#     # Call the main function with the audio data and sampling rate
#     print(data.shape)
#     main(data, sampling_rate)


# import soundfile as sf
# import numpy as np

# def encode_a_law(x, A=87.7):
#     return np.sign(x) * (np.log(1 + A * np.abs(x)) / np.log(1 + A))

# def decode_a_law(y, A=87.7):
#     return np.sign(y) * (1 / A) * ((1 + A) ** np.abs(y) - 1)

# def encode_mu_law(x, mu=256):
#     return np.sign(x) * (np.log(1 + mu * np.abs(x)) / np.log(1 + mu))

# def decode_mu_law(y, mu=256):
#     return np.sign(y) * (1 / mu) * ((1 + mu) ** np.abs(y) - 1)

# def process_wav_file(input_file, output_file, encoding_function, decoding_function):
#     """Applies non-linear PCM encoding/decoding to a WAV file."""
#     data, samplerate = sf.read(input_file)
#     audio_data = data.astype(np.float64)
#     num_channels = audio_data.shape[1] if len(audio_data.shape) > 1 else 1
#     processed_data = np.zeros_like(audio_data, dtype=np.float64)

#     for i in range(len(audio_data)):
#         for channel in range(num_channels):
#             normalized_sample = audio_data[i, channel] / 32768.0
#             encoded_sample = encoding_function(normalized_sample)
#             decoded_sample = decoding_function(encoded_sample)
#             processed_data[i, channel] = decoded_sample * 32767.0

#     sf.write(output_file, processed_data.astype(np.int16), samplerate)

# def plot_wav_file(wav_file, subplot_axes, title):
#     """Plots both channels of a WAV file on given subplot axes."""
#     data, samplerate = sf.read(wav_file)
#     signal = data.astype(np.float64)  # Convert to float64 for plotting
#     num_channels = signal.shape[1] if len(signal.shape) > 1 else 1
#     time = np.linspace(0, len(signal) / samplerate, num=len(signal))

#     for channel in range(num_channels):
#         subplot_axes[channel].plot(time, signal[:, channel] if num_channels > 1 else signal)
#         subplot_axes[channel].set_title(f"{title} - Channel {channel+1}")
#         subplot_axes[channel].set_xlabel("Time (s)")
#         subplot_axes[channel].set_ylabel("Amplitude")

# # Example usage:
# input_wav_file = "src/level2/volume_multiplied.wav"
# output_mu_law_file = "src/level3/output_mu_law.wav"
# output_a_law_file = "src/level3/output_a_law.wav"

# process_wav_file(input_wav_file, output_mu_law_file, encode_mu_law, decode_mu_law)
# process_wav_file(input_wav_file, output_a_law_file, encode_a_law, decode_a_law)

# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(3, 2, figsize=(15, 10))
# plot_wav_file(input_wav_file, axes[0], "Original Waveform")
# plot_wav_file(output_mu_law_file, axes[1], "μ-law Compressed")
# plot_wav_file(output_a_law_file, axes[2], "A-law Compressed")

# plt.tight_layout()
# plt.show()


# import numpy as np
# import soundfile as sf
# import matplotlib.pyplot as plt

# # A-law encoding and decoding
# def encode_a_law(x, A=87.7):
#     abs_x = np.abs(x)
#     encoded = np.where(abs_x < 1 / A,
#                        A * abs_x / (1 + np.log(A)),
#                        (1 + np.log(A * abs_x)) / (1 + np.log(A)))
#     return np.sign(x) * encoded

# def decode_a_law(y, A=87.7):
#     abs_y = np.abs(y)
#     decoded = np.where(abs_y < 1 / (1 + np.log(A)),
#                        abs_y * (1 + np.log(A)) / A,
#                        (np.exp(abs_y * (1 + np.log(A)) - 1)) / A)
#     return np.sign(y) * decoded

# # μ-law encoding and decoding
# def encode_mu_law(x, mu=255):
#     return np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)

# def decode_mu_law(y, mu=255):
#     return np.sign(y) * (1 / mu) * ((1 + mu) ** np.abs(y) - 1)

# # Function to process WAV file
# def process_wav_file(input_file, output_file, encoding_function, decoding_function):
#     data, samplerate = sf.read(input_file)
#     audio_data = data.astype(np.float64)
#     num_channels = audio_data.shape[1] if len(audio_data.shape) > 1 else 1
#     processed_data = np.zeros_like(audio_data, dtype=np.float64)

#     for i in range(len(audio_data)):
#         if num_channels > 1:
#             for channel in range(num_channels):
#                 normalized_sample = audio_data[i, channel] / 32767.0
#                 encoded_sample = encoding_function(normalized_sample)
#                 decoded_sample = decoding_function(encoded_sample)
#                 processed_data[i, channel] = decoded_sample * 32767.0
#         else:
#             normalized_sample = audio_data[i] / 32767.0
#             encoded_sample = encoding_function(normalized_sample)
#             decoded_sample = decoding_function(encoded_sample)
#             processed_data[i] = decoded_sample * 32767.0

#     sf.write(output_file, processed_data.astype(np.int16), samplerate)

# # Function to plot WAV file
# def plot_wav_file(wav_file, ax, title):
#     data, samplerate = sf.read(wav_file)
#     signal = data.astype(np.float64)
#     num_channels = signal.shape[1] if len(signal.shape) > 1 else 1
#     time = np.linspace(0, len(signal) / samplerate, num=len(signal))

#     for channel in range(num_channels):
#         ax.plot(time, signal[:, channel] if num_channels > 1 else signal)
#         ax.set_title(f"{title} - Channel {channel + 1}")
#         ax.set_xlabel("Time (s)")
#         ax.set_ylabel("Amplitude")

# # Example usage
# input_wav_file = "src/level2/volume_multiplied.wav"
# output_mu_law_file = "src/level3/output_mu_law.wav"
# output_a_law_file = "src/level3/output_a_law.wav"

# # Process the files with μ-law and A-law
# process_wav_file(input_wav_file, output_mu_law_file, encode_mu_law, decode_mu_law)
# process_wav_file(input_wav_file, output_a_law_file, encode_a_law, decode_a_law)

# # Plotting
# fig, axes = plt.subplots(3, 1, figsize=(15, 10))
# plot_wav_file(input_wav_file, axes[0], "Original Waveform")
# plot_wav_file(output_mu_law_file, axes[1], "μ-law Processed")
# plot_wav_file(output_a_law_file, axes[2], "A-law Processed")

# plt.tight_layout()
# plt.show()


import numpy as np
from scipy.io import wavfile

def a_law_encode(x, A=87.6):
    """A-law encoding of a signal."""
    sign = np.sign(x)
    x = np.abs(x)
    y = sign * (A * np.log1p(x / A)) / np.log1p(A)
    y = np.clip(y, -1, 1)  # Clip to [-1, 1] range
    y = (y + 1) / 2 * 255  # Bias and scale to [0, 255] range
    return np.round(y).astype(np.int8)

def a_law_decode(y, A=87.6):
    """A-law decoding of a signal."""
    y = y.astype(np.float32) / 255  # Scale to [-1, 1] range
    y = 2 * y - 1  # Unbias
    x = np.sign(y) * A * (np.expm1(np.abs(y) * np.log1p(A))) / A
    return x



def mu_law_encode(x, mu=255):
    """Mu-law encoding of a signal."""
    sign = np.sign(x)
    x = np.abs(x)
    x_mu = x * mu
    log_x_mu = np.log1p(x_mu) / np.log1p(mu)
    y = sign * ((1 + mu) ** log_x_mu - 1) / mu
    y = np.clip(y, -1, 1)  # Clip to [-1, 1] range
    y = (y + 1) / 2 * mu  # Bias and scale to [0, mu] range
    return np.round(y).astype(np.int8)

def mu_law_decode(y, mu=255):
    """Mu-law decoding of a signal."""
    y = y.astype(np.float32)
    y = 2 * y / mu - 1  # Unbias and scale to [-1, 1] range
    y = np.sign(y) * (1 / mu) * ((1 + mu) ** np.abs(y) - 1)
    return y


def process_wav(input_file, output_file, companding_type='a-law'):
    """
    Applies A-law or Mu-law companding to a stereo WAV file,
    handling different input data types.
    """
    # Read input file
    sample_rate, data = wavfile.read(input_file)

    # Store original dtype for later
    original_dtype = data.dtype

    # Normalize to float32 range [-1, 1] if not already float32
    if original_dtype != np.float64:
        data_max = np.iinfo(original_dtype).max
        data = data.astype(np.float64) / data_max
    else:
        data_max = 1.0

    # Apply companding
    if companding_type == 'a-law':
        encoded_data = a_law_encode(data)
    elif companding_type == 'mu-law':
        encoded_data = mu_law_encode(data)
    else:
        raise ValueError("Invalid companding type. Choose 'a-law' or 'mu-law'.")

    # Decode the compressed data
    if companding_type == 'a-law':
        decoded_data = a_law_decode(encoded_data)
    elif companding_type == 'mu-law':
        decoded_data = mu_law_decode(encoded_data)

    # Convert back to the original data type or keep as float32
    if original_dtype != np.float32:
        decoded_data = (decoded_data * data_max).astype(original_dtype)
    else:
        decoded_data = decoded_data.astype(np.float32)

    # Write output file
    wavfile.write(output_file, sample_rate, decoded_data)

# Example usage
process_wav("src/level2/volume_multiplied.wav", "src/level3/output_a_law.wav", companding_type='a-law')
process_wav("src/level2/volume_multiplied.wav", "src/level3/output_mu_law.wav", companding_type='mu-law')
