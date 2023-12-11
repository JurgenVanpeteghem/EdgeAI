import numpy as np

# Set the seed value for experiment reproducibility.
seed = 42
np.random.seed(seed)

def stft(waveform, frame_length=255, frame_step=128):
    # Compute Short-Time Fourier Transform (STFT)
    window = np.hanning(frame_length)
    frames = np.lib.stride_tricks.sliding_window_view(waveform, window_shape=(frame_length,), step=frame_step)
    stft_result = np.fft.fft(frames * window, axis=-1)
    return np.abs(stft_result)

def get_spectrogram(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = np.zeros(16000 - len(waveform), dtype=np.float32)
    # Concatenate the waveform with zero_padding, ensuring all audio clips are of the same length.
    equal_length = np.concatenate([waveform, zero_padding])

    # Convert the waveform to a spectrogram via STFT.
    spectrogram = stft(equal_length)

    # Add a `channels` dimension.
    spectrogram = spectrogram[..., np.newaxis]
    
    return spectrogram

def preprocess_audiobuffer(waveform):
    """
    waveform: ndarray of size (16000, )
    
    output: Spectrogram Tensor of size: (1, `height`, `width`, `channels`)
    """
    # Normalize from [-32768, 32767] to [-1, 1]
    waveform = waveform / 32768.0

    # Get the spectrogram.
    spectrogram = get_spectrogram(waveform)

    # Add one dimension to match the expected input shape.
    spectrogram = spectrogram[np.newaxis, ...]

    return spectrogram
