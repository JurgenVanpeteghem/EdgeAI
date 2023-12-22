import numpy as np
import librosa
from scipy import ndimage

# Set the seed value for experiment reproducibility.
seed = 42
np.random.seed(seed)

def get_spectrogram(waveform):
    spectrograms = []

    # Generate spectrogram using librosa
    spectrogram = librosa.feature.melspectrogram(y=waveform, sr=16000)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Resize spectrogram to a fixed shape without TensorFlow
    target_shape = (128, 128)
    spectrogram = ndimage.zoom(
        spectrogram, (target_shape[0] / spectrogram.shape[0], target_shape[1] / spectrogram.shape[1]))

    # Ensure the shape matches the target_shape
    if spectrogram.shape != target_shape:
        spectrogram = spectrogram[:target_shape[0], :target_shape[1]]

    # Add a channel dimension
    spectrogram = np.expand_dims(spectrogram, axis=-1)

    spectrograms.append(spectrogram)

    spectrograms = np.array(spectrograms)

    return spectrograms
