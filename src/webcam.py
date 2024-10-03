import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import time
import pygame  

pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('C:/Users/alarm_sounds.mp3')  

# Load model untuk deteksi emosi
emotion_model = load_model('models/rnn_model.h5')

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)

face_sequence = []
time_steps = 30 

eye_closed_time = 0
is_eye_closed = False 
alarm_played = False  

def detect_snap(hand_landmarks):
    thumb_tip = hand_landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return abs(thumb_tip.x - index_tip.x) < 0.05 and abs(thumb_tip.y - index_tip.y) < 0.05

# Fungsi untuk memeriksa apakah mata tertutup
def check_eye_closed(landmarks):
    left_eye_height = landmarks[145].y - landmarks[159].y
    right_eye_height = landmarks[374].y - landmarks[386].y
    return left_eye_height < 0.02 and right_eye_height < 0.02

while True:
    ret, frame = cam.read()
    if not ret:
        print("Tidak dapat mengambil frame.")
        break

    # Ubah gambar menjadi RGB dan deteksi wajah
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5).process(frame_rgb)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Deteksi wajah
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
        face_roi = gray_frame[y:y + h, x:x + w]
        face_roi_rgb = cv2.resize(face_roi, (48, 48)) / 255.0
        face_sequence.append(np.repeat(face_roi_rgb[..., np.newaxis], 3, axis=-1))

        if len(face_sequence) > time_steps:
            face_sequence.pop(0)

        # Prediksi emosi jika cukup frame
        if len(face_sequence) == time_steps:
            input_sequence = np.expand_dims(face_sequence, axis=0)
            predictions = emotion_model.predict(input_sequence)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class] * 100
            emotion_label = ["angry", "fear", "happy"][predicted_class]
            cv2.putText(frame, f"{emotion_label}: {confidence:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Deteksi mata tertutup
    if results_face_mesh.multi_face_landmarks:
        for landmarks in results_face_mesh.multi_face_landmarks:
            if check_eye_closed(landmarks.landmark):
                if not is_eye_closed:
                    is_eye_closed = True  
                    eye_closed_time = time.time()
            else:
                # Mata terbuka
                if is_eye_closed:
                    alarm_sound.stop() 
                is_eye_closed = False
                alarm_played = False  

            # Periksa apakah mata tertutup sudah lebih dari 5 detik
            if is_eye_closed and (time.time() - eye_closed_time) > 5:
                cv2.putText(frame, "Kamu Mengantuk!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                
                # Mainkan suara alarm jika belum dimainkan
                if not alarm_played:
                    alarm_sound.play()  
                    alarm_played = True 

    results_hands = hands.process(frame_rgb)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if detect_snap(hand_landmarks.landmark):
                print("Jari dijentikkan!")
                cam.release()
                cv2.destroyAllWindows()
                exit() 

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()