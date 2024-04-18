import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import base64

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

mtcnn_detector = MTCNN()

# Load the model
model_path = 'gender_classification_model.h5'
try:
    loaded_model = tf.keras.models.load_model(model_path)
except Exception as e:
    print("Error loading model:", str(e))
    exit()

uploaded_image = None


def preprocess_image(base64_img, target_size=(50, 50)):
    try:
        if base64_img is None:
            raise ValueError("Image data is None")

        image_data = base64.b64decode(base64_img)
        img_np = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image data")

        if img.size == 0:
            raise ValueError("Empty image data")

        cv2.imwrite('uploaded_image.jpg', img)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        return img
    except Exception as e:
        print("Error preprocessing image:", str(e))
        return None


def detect_faces(image):
    try:
        faces = mtcnn_detector.detect_faces(image)
        faces_rect = [(face['box'][0], face['box'][1], face['box'][2], face['box'][3]) for face in faces]
        return faces_rect
    except Exception as e:
        print("Error detecting faces:", str(e))
        return None


def is_face_incomplete(face, image_width, image_height):
    x, y, w, h = face
    # Check if the face width or height is less than a threshold (e.g., half of the image size)
    if w < 0.5 * image_width or h < 0.5 * image_height:
        return True
    return False


def predict_gender(img, faces, image_width, image_height, threshold=0.5):
    try:
        if faces is None:
            print("No faces detected in the image.")
            return

        gender_predictions = []

        for face in faces:
            if is_face_incomplete(face, image_width, image_height):
                print("Incomplete face detected. Please re-enter the image.")
                return None

            x, y, w, h = face
            face_img = img[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (50, 50))
            face_img = face_img / 255.0
            face_img = np.expand_dims(face_img, axis=0)

            predictions = loaded_model.predict(face_img)
            predicted_class = 1 if predictions[0][0] > threshold else 0
            gender_label = "Male" if predicted_class == 1 else "Female"
            print("Predicted gender for face:", gender_label)
            gender_predictions.append(gender_label)

        return gender_predictions

    except Exception as e:
        print("Error predicting gender:", str(e))
        return None


@app.route('/api', methods=['PUT'])
def index():
    global uploaded_image
    try:
        input_data = request.get_data()
        print("Received data:", input_data)  # Debugging print statement
        uploaded_image = preprocess_image(input_data)
        if uploaded_image is None:
            return jsonify({'error': 'Failed to preprocess uploaded image'}), 400
        return jsonify({'message': 'Image uploaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['GET'])
def predict():
    try:
        global uploaded_image
        if uploaded_image is None:
            return jsonify({'error': 'No image uploaded'}), 400
        img = cv2.imread('uploaded_image.jpg')
        image_height, image_width, _ = img.shape
        faces = detect_faces(img)
        gender_predictions = predict_gender(img, faces, image_width, image_height)
        if gender_predictions:
            return jsonify({'gender_predictions': gender_predictions})
        else:
            return jsonify({'error': 'Incomplete face detected. Please re-enter the image.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=80)
