import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from keras import layers
from keras import models
#from IPython import display

import wave
DATASET_PATH = 'data/mini_speech_commands'

data_dir = pathlib.Path(DATASET_PATH)

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
print('Commands:', commands)

def get_spectrogram(waveform):
   input_len = 16000
   waveforms = waveform[:input_len]
   zero_padding = tf.zeros( [16000] - tf.shape(waveforms), dtype=tf.float32)
   waveforms = tf.cast(waveforms,dtype=tf.float32)
   equal_length = tf.concat([waveforms,zero_padding], 0)
   spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
   spectrogram = tf.abs(spectrogram)
   spectrogram = spectrogram[... , tf.newaxis]
   return spectrogram


sample_file = data_dir/'stop/0b40aa8e_nohash_0.wav'

obj = wave.open(str(sample_file), 'rb')
n_samples = obj.getnframes()
signal_wave = obj.readframes(n_samples)
signal_array = np.frombuffer(signal_wave, dtype=np.int16)
obj.close()

print(signal_array.shape)

loaded_model = models.load_model("saved_model")

# set in correct shape
waveform = signal_array / 32768
waveform = tf.convert_to_tensor(waveform,dtype= tf.float32)
spec = get_spectrogram(waveform)
spec = tf.expand_dims(spec, 0)
prediction = loaded_model(spec)
print(prediction)

label_pred = np.argmax(prediction, axis=1)
print(commands[label_pred[0]])
plt.bar(commands, tf.nn.softmax(prediction[0]))
plt.title(f'Predictions for "left"')
plt.show()