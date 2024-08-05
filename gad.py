import streamlit as st
import cv2
import numpy as np
from fer import FER
import os
import tensorflow as tf

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Load models
faceProto = "C:/Users/kunwa/Python/Projects/gad/opencv_face_detector.pbtxt"
faceModel = "C:/Users/kunwa/Python/Projects/gad/opencv_face_detector_uint8.pb"
ageProto = "C:/Users/kunwa/Python/Projects/gad/age_deploy.prototxt"
ageModel = "C:/Users/kunwa/Python/Projects/gad/age_net.caffemodel"
genderProto = "C:/Users/kunwa/Python/Projects/gad/gender_deploy.prototxt"
genderModel = "C:/Users/kunwa/Python/Projects/gad/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
emotion_detector = FER()

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

# Example usage of tf.compat.v1.reset_default_graph
tf.compat.v1.reset_default_graph()

def detect_age_gender_emotion(frame):
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        return resultImg, [], []

    predictions = []
    padding = 20
    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                     max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        emotion, score = emotion_detector.top_emotion(face)

        predictions.append({'gender': gender, 'age': age[1:-1], 'emotion': emotion})

    return resultImg, faceBoxes, predictions

# Streamlit app
st.title("Age, Gender, and Emotion Detection")

# S

# Add a header image
header_image = "C:/Users/kunwa/Python/Projects/gad/cover.png"
st.image(header_image, use_column_width=True)

# Add a sidebar with options
option = st.sidebar.selectbox("Choose input source", ("Upload Image", "Use Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        # Detect age, gender, and emotion
        resultImg, faceBoxes, predictions = detect_age_gender_emotion(frame)

        # Display the image with detections
        st.image(resultImg, channels="BGR")

        # Display predictions
        for i, prediction in enumerate(predictions):
            st.write(f"Person {i+1}: Gender - {prediction['gender']}, Age - {prediction['age']}, Emotion - {prediction['emotion']}")

elif option == "Use Webcam":
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect age, gender, and emotion
        resultImg, faceBoxes, predictions = detect_age_gender_emotion(frame)

        # Overlay predictions on the image
        for i, prediction in enumerate(predictions):
            faceBox = faceBoxes[i]
            label = f"{prediction['gender']}, {prediction['age']}, {prediction['emotion']}"
            cv2.putText(resultImg, label, (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the image with detections and predictions
        FRAME_WINDOW.image(resultImg, channels="BGR")

    cap.release()