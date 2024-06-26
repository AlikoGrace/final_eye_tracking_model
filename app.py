from flask import Flask, request, jsonify
import cv2
import numpy as np
import dlib
import os
from imutils import face_utils
import pandas as pd
import sys
import logging

app = Flask(__name__)

# Load dlib's pre-trained face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        logger.error("No video file provided")
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({"error": "No selected file"}), 400
    
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)
    
    try:
        # Process the video
        cap = cv2.VideoCapture(filepath)
        frame_counter = 0
        right_eye_progression_x = []
        right_eye_progression_y = []
        left_eye_progression_x = []
        left_eye_progression_y = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                frame_counter += 1
                ReLcX = ReLcY = ReRcX = ReRcY = LeLcX = LeLcY = LeRcX = LeRcY = 0

                for i, (x, y) in enumerate(shape):
                    if i == 42:  # Right eye left corner
                        ReLcX, ReLcY = x, y
                    elif i == 45:  # Right eye right corner
                        ReRcX, ReRcY = x, y
                    elif i == 36:  # Left eye left corner
                        LeLcX, LeLcY = x, y
                    elif i == 39:  # Left eye right corner
                        LeRcX, LeRcY = x, y

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

        fixation_results = analyze_fixations(right_eye_progression_x, right_eye_progression_y, left_eye_progression_x, left_eye_progression_y)
        logger.info("Video processed successfully")
        return jsonify(fixation_results)

    except Exception as e:
        logger.error("Error processing video: %s", str(e))
        return jsonify({"error": str(e)}), 500



    

def analyze_fixations(rx, ry, lx, ly):
    fixation_frequency = len(rx)
    average_fixation_duration = sum(rx) / len(rx) if rx else 0
    
    # Implement actual logic based on research paper
    threshold = 500  # Example threshold, adjust based on your research
    prediction = "likely dyslexic" if average_fixation_duration > threshold else "not dyslexic"

    return {
        "fixation_frequency": fixation_frequency,
        "average_fixation_duration": average_fixation_duration,
        "prediction": prediction
    }


if __name__ == '__main__':
    app.run(debug=True)
