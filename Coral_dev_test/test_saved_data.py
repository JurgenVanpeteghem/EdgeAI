import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

import wave
import pyaudio

from keras import layers
from keras import models
#from IPython import display

interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

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

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
audio = pyaudio.PyAudio()

def record_audio():
   stream = audio.open(
      format=FORMAT,
      channels=CHANNELS,
      rate=RATE,
      input=True,
      input_device_index=0,
      frames_per_buffer=FRAMES_PER_BUFFER
   )

   print("start recording...")

   frames = []
   seconds = 1
   for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
      data = stream.read(FRAMES_PER_BUFFER)
      frames.append(data)

   print("recording stopped")

   stream.stop_stream()
   stream.close()

   # Save the recorded audio as a WAV file
   with wave.open("test.wav", 'wb') as wf:
      wf.setnchannels(CHANNELS)
      wf.setsampwidth(audio.get_sample_size(FORMAT))
      wf.setframerate(RATE)
      wf.writeframes(b''.join(frames))


while True:
   if input():
      record_audio()
      # sample_file = data_dir/'stop/0b40aa8e_nohash_0.wav'
      sample_file = 'test.wav'

      obj = wave.open(str(sample_file), 'rb')
      n_samples = obj.getnframes()
      signal_wave = obj.readframes(n_samples)
      signal_array = np.frombuffer(signal_wave, dtype=np.int16)
      obj.close()

      print(signal_array.shape)

      # set in correct shape
      waveform = signal_array / 32768
      waveform = tf.convert_to_tensor(waveform,dtype= tf.float32)
      spec = get_spectrogram(waveform)
      spec = tf.expand_dims(spec, 0)
      
      input_tensor = interpreter.tensor(interpreter.get_input_details()[0]['index'])
      output_tensor = interpreter.tensor(interpreter.get_output_details()[0]['index'])

      input_tensor()[0] = spec
      interpreter.invoke()

      prediction = output_tensor()[0]
      label_pred = np.argmax(prediction)
      
      print("prediction:", commands[label_pred])
      plt.bar(commands, tf.nn.softmax(prediction[0]))
      plt.title(f'Predictions for "left"')
      plt.show()
