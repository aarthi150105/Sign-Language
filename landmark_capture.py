import cv2
import os
import time
import numpy as np
import mediapipe as mp

SAVE_LIMIT = 500
LANDMARK_DIM = 63  # 21 landmarks * 3 (x, y, z)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def collect_landmarks(class_name):
    save_dir = os.path.join("landmarks_dataset", class_name)
    os.makedirs(save_dir, exist_ok=True)

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Webcam not accessible.")
            return

        count = 0
        print(f"\n🖐️ Starting landmark collection for class '{class_name}'...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])

                    if len(landmarks) == LANDMARK_DIM:
                        npy_path = os.path.join(save_dir, f"{class_name}_{count+1}.npy")
                        np.save(npy_path, np.array(landmarks))
                        count += 1
                        print(f"✅ Saved {count}/{SAVE_LIMIT}", end="\r")
                        time.sleep(0.03)

            cv2.putText(frame, f"{class_name}: {count}/{SAVE_LIMIT}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Collecting Landmarks", frame)

            if cv2.waitKey(1) == 27 or count >= SAVE_LIMIT:
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Done collecting {count} landmark samples for '{class_name}'.")

# Run
if __name__ == "__main__":
    while True:
        class_name = input("Enter the class name for this gesture: ").strip().lower()
        if not class_name:
            print("⚠️ Class name cannot be empty.")
            continue

        collect_landmarks(class_name)

        again = input("Do you want to collect another class? (yes/no): ").strip().lower()
        if again != 'yes':
            print("🎉 Landmark data collection complete!")
            break
