import numpy as np
import tflite_runtime.interpreter as tflite
import time
from recording_helper import record_audio, terminate
from tf_helper import preprocess_audiobuffer
from periphery import GPIO

# !! Modify this in the correct order
commands = ['left', 'down', 'stop', 'up', 'right', 'no', 'go', 'yes']
#commands = ['five', 'four', 'off', 'on', 'stop', 'three', 'yes', 'zero', 'blabla']

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
    spec = get_spectrogram(sample_file)

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
            if input():
                command = predict_mic()
                print(command)
                if command == "stop":
                    terminate()
                    break
            if command == "left":
                led1.write(True)
                time.sleep(2)
                led1.write(False)
            elif command == "down":
                led2.write(True)
                time.sleep(2)
                led2.write(False)
            elif command == "right":
                led3.write(True)
                time.sleep(2)
                led3.write(False)
    except KeyboardInterrupt:
        pass
    finally:
        terminate()
