import numpy as np
import tflite_runtime.interpreter as tflite
import time
import wave
from recording_helper import record_audio, terminate
from tf_helper import get_spectogram
from periphery import GPIO

# !! Modify this in the correct order
#commands = ['left', 'down', 'stop', 'up', 'right', 'no', 'go', 'yes']
#commands = ['five', 'four', 'off', 'on', 'stop', 'three', 'yes', 'zero']
commands = ['drie', 'een', 'klaar', 'licht', 'stop', 'uit']

# leds (use python-periphery)
# python3 -m pip install python-periphery
led1 = GPIO("/dev/gpiochip2", 13, "out")  # pin 37
led2 = GPIO("/dev/gpiochip4", 13, "out")  # pin 36
led3 = GPIO("/dev/gpiochip0", 8, "out")  # pin 31

# Load the TensorFlow Lite model.
interpreter = tflite.Interpreter(model_path="keyword_recognition_model.tflite")
interpreter.allocate_tensors()

# Print the expected input shape
input_shape = interpreter.get_input_details()[0]['shape']
#print("Expected Input Shape:", input_shape)

# Get input and output tensors.
input_tensor = interpreter.tensor(interpreter.get_input_details()[0]['index'])
output_tensor = interpreter.tensor(interpreter.get_output_details()[0]['index'])

def predict_mic():
    audio = record_audio()
    sample_file = 'recorded.wav'
    
    obj = wave.open(str(sample_file), 'rb')
    n_samples = obj.getnframes()
    signal_wave = obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)
    obj.close()

    waveform = signal_array / 32768
    spec = get_spectogram(waveform)

    # Set input tensor.
    input_tensor()[0] = spec

    # Run inference.
    interpreter.invoke()

    # Get the output.
    prediction = output_tensor()[0]
    label_pred = np.argmax(prediction)

    command = commands[label_pred]
    print("Predicted label:", command)
    return command

if __name__ == "__main__":
    try:
        while True:
            if input() == "":
                command = predict_mic()
                print(command)
            if command == "drie":
                led1.write(True)
                time.sleep(2)
                led1.write(False)
            elif command == "een":
                led2.write(True)
                time.sleep(2)
                led2.write(False)
            elif command == "stop":
                led3.write(True)
                time.sleep(2)
                led3.write(False)
    except KeyboardInterrupt:
        pass
    finally:
        terminate()
