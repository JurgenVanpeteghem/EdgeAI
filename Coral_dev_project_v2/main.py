import numpy as np
import tflite_runtime.interpreter as tflite
import time
import wave
from recording_helper import record_audio, terminate
from tf_helper import get_spectogram

# keywords
commands = ['drie', 'een', 'klaar', 'licht', 'stop', 'uit']

# Load the TensorFlow Lite model.
# interpreter = tflite.Interpreter(model_path="keyword_recognition_model_full_data.tflite")
interpreter = tflite.Interpreter(model_path="keyword_recognition_model_full_data_edgetpu.tflite")
interpreter.allocate_tensors()

# Print the expected input shape
input_shape = interpreter.get_input_details()[0]['shape']
#print("Expected Input Shape:", input_shape)

# Get input and output tensors.
input_tensor = interpreter.tensor(interpreter.get_input_details()[0]['index'])
output_tensor = interpreter.tensor(interpreter.get_output_details()[0]['index'])

def export_gpio(pin):
    try:
        with open('/sys/class/gpio/export', 'w') as f:
            f.write(str(pin))
    except IOError:
        print("Error exporting GPIO pin {}".format(pin))

def set_direction(pin, direction):
    try:
        with open('/sys/class/gpio/gpio{}/direction'.format(pin), 'w') as f:
            f.write(direction)
    except IOError:
        print("Error setting direction for GPIO pin {}".format(pin))

def write_gpio(pin, value):
    try:
        with open('/sys/class/gpio/gpio{}/value'.format(pin), 'w') as f:
            f.write(str(value))
    except IOError:
        print("Error writing value to GPIO pin {}".format(pin))

# Configure GPIO pins
led1_pin = 77  # GPIO pin for led1
led2_pin = 141  # GPIO pin for led2
led3_pin = 8  # GPIO pin for led3

# Export GPIO pins
export_gpio(led1_pin)
export_gpio(led2_pin)
export_gpio(led3_pin)

# Set GPIO pin directions
set_direction(led1_pin, 'out')
set_direction(led2_pin, 'out')
set_direction(led3_pin, 'out')

# write all leds low
write_gpio(led1_pin, 0)
write_gpio(led2_pin, 0)
write_gpio(led3_pin, 0)

def is_audio_spoken(audio_data):
    return True

def predict_mic():
    audio = record_audio()
    print(is_audio_spoken(audio))
    if not is_audio_spoken(audio):
        return None, 0
    
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
    confidence = prediction[label_pred]
    

    command = commands[label_pred]
    return command, confidence

if __name__ == "__main__":
    try:
        while True:
            if input() == "":
                command, confidence = predict_mic()
                if command is not None and confidence > 0.65:
                    print("Predicted keyword:", command)
                
                    if command in ['drie', 'een', 'klaar', 'licht', 'stop', 'uit']:
                        if command == 'een':
                            write_gpio(led1_pin, 1)
                        elif command == "licht":
                            write_gpio(led2_pin, 1)
                        elif command == "drie":
                            write_gpio(led3_pin, 1)
                        elif command == 'klaar':
                            write_gpio(led1_pin, 0)
                        elif command == 'uit':
                            write_gpio(led2_pin, 0)
                        elif command == 'stop':
                            write_gpio(led3_pin, 0)
    except KeyboardInterrupt:
        pass
    finally:
        # Unexport GPIO pins when done
        try:
            with open('/sys/class/gpio/unexport', 'w') as f:
                f.write(str(led1_pin))
                f.write(str(led2_pin))
                f.write(str(led3_pin))
        except IOError:
            print("Error unexporting GPIO pins")
        terminate()
