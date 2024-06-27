from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh

def create_line_iterator(P1, P2, img):
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X, P1Y = P1
    P2X, P2Y = P2
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
    itbuffer.fill(np.nan)
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:
        itbuffer[:, 0] = P1X
        if negY:
            itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
    elif P1Y == P2Y:
        itbuffer[:, 1] = P1Y
        if negX:
            itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
        else:
            itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
    else:
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32) / dY.astype(np.float32)
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(int) + P1X
        else:
            slope = dY.astype(np.float32) / dX.astype(np.float32)
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(int) + P1Y
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]
    itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]
    return itbuffer

@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files['video']
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    # Process the video
    cap = cv2.VideoCapture(filepath)
    frame_counter = 0
    right_eye_progression_x = []
    right_eye_progression_y = []
    left_eye_progression_x = []
    left_eye_progression_y = []

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    frame_counter += 1

                    # Get eye landmarks
                    ReLcX, ReLcY = int(face_landmarks.landmark[133].x * frame.shape[1]), int(face_landmarks.landmark[133].y * frame.shape[0])
                    ReRcX, ReRcY = int(face_landmarks.landmark[33].x * frame.shape[1]), int(face_landmarks.landmark[33].y * frame.shape[0])
                    LeLcX, LeLcY = int(face_landmarks.landmark[362].x * frame.shape[1]), int(face_landmarks.landmark[362].y * frame.shape[0])
                    LeRcX, LeRcY = int(face_landmarks.landmark[263].x * frame.shape[1]), int(face_landmarks.landmark[263].y * frame.shape[0])

                    right_eye_center_x = (ReLcX + ReRcX) // 2
                    right_eye_center_y = (ReLcY + ReRcY) // 2
                    left_eye_center_x = (LeLcX + LeRcX) // 2
                    left_eye_center_y = (LeLcY + LeRcY) // 2

                    right_eye_progression_x.append(right_eye_center_x)
                    right_eye_progression_y.append(right_eye_center_y)
                    left_eye_progression_x.append(left_eye_center_x)
                    left_eye_progression_y.append(left_eye_center_y)

    cap.release()
    os.remove(filepath)

    # Analyze fixation frequency and duration
    fixation_results = analyze_fixations(right_eye_progression_x, right_eye_progression_y, left_eye_progression_x, left_eye_progression_y)
    
    return jsonify(fixation_results)

def analyze_fixations(rx, ry, lx, ly):
    # Placeholder logic for analysis
    fixation_frequency = len(rx)
    average_fixation_duration = sum(rx) / len(rx) if rx else 0
    
    # Implement actual logic based on research paper
    threshold = 5  # Example threshold, replace with actual logic
    prediction = "likely dyslexic" if average_fixation_duration > threshold else "not dyslexic"

    return {
        "fixation_frequency": fixation_frequency,
        "average_fixation_duration": average_fixation_duration,
        "prediction": prediction
    }

if __name__ == '__main__':
    app.run(debug=True)
