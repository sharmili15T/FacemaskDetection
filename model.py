from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained mask detection model
model = load_model('path/to/your/mask_detection_model.h5')

# Function to process the image for mask detection
def process_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        resized = cv2.resize(face_roi, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))
        result = model.predict(reshaped)

        label = "Mask" if result[0][0] > 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    return img

# Flask route to handle image upload and display the result
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', message='No selected file')

        # Save the uploaded image
        image_path = 'static/uploaded_image.jpg'
        file.save(image_path)

        # Process the image for mask detection
        result_image = process_image(image_path)

        return render_template('index.html', message='File uploaded successfully', image_path=image_path)

    return render_template('index.html', message='Upload an image')

if __name__ == '__main__':
    app.run(debug=True)
