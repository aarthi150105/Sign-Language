import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os
import pyttsx3
import threading
import time
import queue
import pickle

# Load the trained model
model = load_model("sign_landmark_model.h5")

# Load the label encoder for consistent class names
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
class_names = le.classes_

# Initialize pyttsx3 TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Set female voice if available
voices = engine.getProperty('voices')
for voice in voices:
    if "female" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break

# Queue for speaking text
speak_queue = queue.Queue()

# Dedicated TTS thread
def tts_worker():
    while True:
        text = speak_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speak_queue.task_done()

# Start TTS thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Start webcam
cap = cv2.VideoCapture(0)
last_spoken = ""
last_time_spoken = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for natural view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks).reshape(1, -1)

            # Predict gesture
            prediction = model.predict(landmarks)
            predicted_index = np.argmax(prediction)
            predicted_label = class_names[predicted_index]
            confidence = prediction[0][predicted_index]

            current_time = time.time()
            if predicted_label != last_spoken and confidence > 0.80 and current_time - last_time_spoken > 2:
                speak_queue.put(predicted_label)
                last_spoken = predicted_label
                last_time_spoken = current_time

            # Display prediction
            text = f"{predicted_label} ({confidence*100:.1f}%)"
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Sign Language Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
speak_queue.put(None)  # Stop TTS thread
tts_thread.join()
