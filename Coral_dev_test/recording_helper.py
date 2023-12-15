import keyboard
import pyaudio  # recording audio
import wave     # saving recorded audio as WAV files

# Global variable to track recording status
recording = False

# Function to record audio from the microphone
def record_audio(filename):

    global recording
    FORMAT = pyaudio.paInt16    # audio format
    CHANNELS = 1                # number of channels
    RATE = 16000                # sample rate
    CHUNK = 1024                # chunk size

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    frames = []

    while recording:
        data = stream.read(CHUNK)
        frames.append(data)

    # Stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio as a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

# Function to start and stop recording
def toggle_recording():
    global recording
    if recording:
        recording = False
        print("Stopped recording")
    else:
        recording = True
        print("Recording...")

# Main function
def main():
    # The audio file where you want to save the recorded audio
    audio_filename = "recorded_audio.wav"

    # Register key events for manual recording start/stop control
    keyboard.add_hotkey("r", toggle_recording)

    while True:
        if recording:
            # Call record_audio when recording is enabledrr
            record_audio(audio_filename)
            
if __name__ == "__main__":
    main()
