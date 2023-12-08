import numpy as np
from keras import models
from recording_helper import record_audio, terminate
from tf_helper import preprocess_audiobuffer
import tflite_runtime.interpreter as tflite
#import 'recorded_audio_test.wav' as tes1

# !! Modify this in the correct order
commands = ['left', 'down', 'stop', 'up', 'right', 'no', 'go', 'yes']

#loaded_model = models.load_model("saved_model")
interpreter = tflite.Interpreter("saved_model")

def predict_mic():
    audio = record_audio()
    spec = preprocess_audiobuffer(audio)
    prediction = loaded_model(spec)
    label_pred = np.argmax(prediction, axis=1)
    command = commands[label_pred[0]]
    print("Predicted label:", command)
    return command

if __name__ == "__main__":
    while True:
        command = predict_mic()
        print(command)
        if command == "stop":
            terminate()
            break
        input()
