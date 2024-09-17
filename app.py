import pymongo  # Library for interacting with MongoDB
import os, uuid  # Libraries for file operations and generating unique IDs
import datetime  # Library for handling date and time
import cv2 as cv  # OpenCV library for image processing
import numpy as np  # Library for numerical operations
import pandas as pd  # Library for data manipulation
import streamlit as st  # Streamlit library for creating web applications
import tensorflow as tf  # TensorFlow library for machine learning
from deepface import DeepFace  # DeepFace library for facial recognition

# Load the pre-trained model for mask classification
model = tf.keras.models.load_model("mask_classification.h5")
model.compile(
    optimizer='adam',  # Adam optimizer for training
    loss='binary_crossentropy',  # Loss function for binary classification
    metrics=[
        tf.keras.metrics.BinaryAccuracy(),  # Metric to track binary accuracy
        tf.keras.metrics.Precision(),  # Metric to track precision
        tf.keras.metrics.Recall(),  # Metric to track recall
        tf.keras.metrics.AUC()  # Metric to track AUC (Area Under the Curve)
    ]
)

# Set up MongoDB connection
client = pymongo.MongoClient("mongodb+srv://admin:admin@facemaskdetection.b9ekp.mongodb.net/")
db = client["facemaskdetection"]
login_collection = db["login"]  # Collection for user login information
logs_collection = db["logs"]  # Collection for storing logs of user activities

def inference_pipeline(
    pixel_array,
    class_dict = {
        0: 'with_mask',
        1: 'without_mask'
    }
):
    # Convert the image from BGR to RGB format
    pixel_array = cv.cvtColor(
        pixel_array, 
        cv.COLOR_BGR2RGB
    )
    # Resize the image to match the input size of the model
    pixel_array = cv.resize(
        pixel_array, 
        (128, 128)
    )
    # Add an extra dimension for the batch size
    pixel_array = np.expand_dims(
        pixel_array, 
        axis=0
    )
    # Predict the class of the image
    prediction = model.predict(pixel_array)
    # Round the prediction to get a binary class label (0 or 1)
    prediction = np.round(prediction).flatten()[0]
    # Map the numeric prediction to a class name
    prediction = class_dict[prediction]
    return prediction

def visualize_face_detection(frame):
    # Save the frame to a temporary file with a unique name
    img_path = f"uploads/{str(uuid.uuid4())}.jpg"
    cv.imwrite(img_path, cv.cvtColor(frame, cv.COLOR_RGB2BGR))

    # Use DeepFace to detect and represent faces in the image
    face_objs = DeepFace.represent(
        img_path = img_path,
        model_name = "Facenet512",
        enforce_detection = False
    )
    # Filter faces based on confidence threshold
    face_objs = [obj for obj in face_objs if obj['face_confidence'] > 0.80]
    try:
        if len(face_objs) == 0:
            # If no faces are detected, use the mask classification model to predict
            prediction = inference_pipeline(cv.cvtColor(frame, cv.COLOR_RGB2BGR))
            prediction = "With Mask" if len(face_objs) == 0 else "Without Mask"
            cv.putText(frame, f"Prediction: {prediction}", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 2)
        else:
            # If faces are detected, annotate each detected face
            for i in range(len(face_objs)):
                facial_area = face_objs[i]['facial_area']
                if facial_area['w'] * facial_area['h'] > 5000:
                    face_confidence = face_objs[i]['face_confidence']
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    face_img = frame[y:y+h, x:x+w]
                    cv.imwrite(img_path, cv.cvtColor(face_img, cv.COLOR_RGB2BGR))

                    prediction = inference_pipeline(cv.cvtColor(face_img, cv.COLOR_RGB2BGR))
                    prediction = "With Mask" if len(face_objs) == 0 else "Without Mask"
                    cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv.putText(frame, f"Confidence: {face_confidence}", (x, y-20), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    cv.putText(frame, f"Prediction: {prediction}", (x, y-50), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return frame, prediction
    except:
        return None, None

# Main function to run the Streamlit app
if __name__ == "__main__":
    placeholder = st.empty()
    # Check if the user is authenticated
    if not 'auth' in st.session_state:
        st.session_state.auth = False

    # Create login and registration form
    with placeholder.form("login"):
        st.markdown("#### Enter your credentials")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        name = st.text_input("Name")
        login_button = st.form_submit_button("Login")
        register_button = st.form_submit_button("Register")

    if register_button:
        # Register new user
        if not email or not password or not name:
            st.error("Please fill all the fields")
        df_login = pd.DataFrame(list(login_collection.find()))
        if len(df_login) > 0:
            if email in df_login["email"].values:
                st.session_state.auth = False
                st.error("Email already exists")
                st.stop()
            login_collection.insert_one({
                "name": name,
                "email": email,
                "password": password
            })
            st.session_state.auth = True
            st.success("Registration successful")
        else:
            login_collection.insert_one({
                "name": name,
                "email": email,
                "password": password
            })
            st.session_state.auth = True
            st.success("Registration successful")

    if login_button:
        # Authenticate user login
        df_login = pd.DataFrame(list(login_collection.find()))
        if len(df_login) > 0:
            if email in df_login["email"].values and password in df_login["password"].values:
                st.session_state.name = str(df_login[df_login["email"] == email]["name"].values[0])
                st.session_state.auth = True
                st.success("Login successful")
            else:
                st.session_state.auth = False
                st.error("Login failed")
        else:
            st.error("No user found")

    # Clear the form after successful authentication
    if st.session_state.auth:
        placeholder.empty()

        # Create a live webcam feed
        st.title("Webcam Live Feed")
        run = st.checkbox('Run')
        FRAME_WINDOW = st.image([])
        camera = cv.VideoCapture(1)

        while run:
            _, frame = camera.read()
            frame = cv.flip(frame, 1)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame, prediction = visualize_face_detection(frame)
            if frame is None:
                pass
            else:
                FRAME_WINDOW.image(frame)
                logs_collection.insert_one({
                    "name": st.session_state.name,
                    "timestamp": datetime.datetime.now(),
                    "prediction": prediction
                })

        else:
            st.write('Stopped')
