import numpy as np
import tflite_runtime.interpreter as tflite
import time
from recording_helper import record_audio, terminate
from tf_helper import preprocess_audiobuffer

# !! Modify this in the correct order
commands = ['left', 'down', 'stop', 'up', 'right', 'no', 'go', 'yes']

# Load the TensorFlow Lite model.
interpreter = tflite.Interpreter(model_path="path/to/your/tflite_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_tensor = interpreter.tensor(interpreter.get_input_details()[0]['index'])
output_tensor = interpreter.tensor(interpreter.get_output_details()[0]['index'])

def predict_mic():
    audio = record_audio()
    spec = preprocess_audiobuffer(audio)

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
            command = predict_mic()
            print(command)
            if command == "stop":
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        terminate()
