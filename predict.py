from re import S
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 128, 128), (255, 165, 0), (0, 128, 0), (255, 192, 203)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        if num >= len(colors):
            break
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
        
    return output_frame


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
    
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

model = keras.models.load_model('mohinh.h5')
actions = np.array(['binh','binh duong','xin chao','toi song o','ten toi la','rat vui duoc gap ban'])

# 1. New detection variables
sequence = []
sentence = [] 
threshold = 0.8
predictions = []

import time

cap = cv2.VideoCapture(0)  # mở webcam

start_recording = False  # biến để bật/tắt quay video
frames = []  # danh sách các frame ảnh từ video đã quay
start_time = 0  # thời điểm bắt đầu quay video
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        # Hiển thị hướng dẫn bấm phím space để bắt đầu quay video
        cv2.putText(frame, 'Press SPACE to start recording', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Xử lý sự kiện bấm phím
        key = cv2.waitKey(1)
        if key == ord('q'):  # Bấm 'q' để thoát
            break 
        elif key == ord(' '):  # Bấm 'SPACE' để bắt đầu/quay lại video
            if not start_recording:
                start_recording = True
                frames = []
                start_time = time.time()  # Lưu thời điểm bắt đầu quay video
        elif start_recording:  # Nếu đang quay video thì lưu frame vào danh sách
            frames.append(frame)
            if len(frames) >= 30*3:  # Nếu đã quay đủ 5s thì dừng quay và xử lý
                start_recording = False
                sequence = []
                for f in frames:
                    image, results = mediapipe_detection(f, holistic)
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-90:]

                if len(sequence) == 90:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    frames = []

        # Hiển thị đồng hồ đếm giây khi bắt đầu quay video
        if start_recording:
            elapsed_time = int(time.time() - start_time)
            cv2.putText(frame, f'Recording... {elapsed_time}s', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Hiển thị ảnh
        cv2.imshow('Webcam', frame)

cap.release()
cv2.destroyAllWindows()

