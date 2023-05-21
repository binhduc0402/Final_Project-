import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils 

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 

def draw_styled_landmarks(image, results):
    
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh   = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh   = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])


# Đường dẫn thư mục chứa các video đầu vào
VIDEOS_PATH = r'videoin3'

# Đường dẫn thư mục chứa các giá trị keypoint 
KEYPOINTS_PATH = r'videoout3'
frame_num1 = 0
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # Loop through input video directories
    for dirpath, dirnames, filenames in os.walk(VIDEOS_PATH):
        # Loop through input video files
        for filename in filenames:
            # Check if file is a video
            if filename.endswith(('.mp4', '.avi', '.mov', '.flv')):
                # Set path for input video file
                video_path = os.path.join(dirpath, filename)
                # Set path for output keypoints directory
                keypoints_dir = os.path.join(KEYPOINTS_PATH, os.path.relpath(dirpath, VIDEOS_PATH), f'{frame_num1}')
                os.makedirs(keypoints_dir, exist_ok=True)
                frame_num1 += 1
                if frame_num1==30:
                    frame_num1=0
                # Open video file
                cap = cv2.VideoCapture(video_path)

                keypoints_all = []  # Lưu trữ keypoints cho tất cả các frame của video

                # Loop through frames of the video
                frame_num = 0
                while cap.isOpened():
                    # Read feed
                    ret, frame = cap.read()

                    # Check if frame was read successfully
                    if not ret:
                        break

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)

                    # Apply wait logic
                    cv2.imshow('OpenCV Feed', image)

                    # Export keypoints
                    keypoints = extract_keypoints(results)
                    keypoints_all.append(keypoints)

                    # Set path for output keypoints file for this frame
                    keypoints_frame_path = os.path.join(keypoints_dir, f'{frame_num}.npy')

                    # Save keypoints to file
                    np.save(keypoints_frame_path, keypoints)

                    frame_num += 1
                    # Break gracefully after 90 frames
                    if frame_num == 90:
                        break
                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break


                cap.release()
                cv2.destroyAllWindows()



