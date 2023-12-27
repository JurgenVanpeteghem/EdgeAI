import pyaudio
import numpy as np

FRAMES_PER_BUFFER = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

def record_audio(frame_size):
    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=0,  # Adjust this index based on your microphone device
        frames_per_buffer=frame_size
    )

    print("Start recording...")

    frames = []
    seconds = 1  # Adjust this duration based on your requirements

    for _ in range(0, int(RATE / frame_size * (seconds * 1000 / FRAMES_PER_BUFFER))):
        data = stream.read(frame_size)
        frames.append(data)

    print("Recording stopped")

    # Convert the frames to a NumPy array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

    stream.stop_stream()
    stream.close()

    # Save the recorded audio as a WAV file
    with wave.open("recorded.wav", 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    return audio_data


def terminate():
    audio.terminate()
