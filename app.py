from tensorflow import keras
import numpy as np
from PIL import Image
import io

from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model = keras.models.load_model('my_model.keras')

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((128, 128))  # Change if your model uses a different size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET'])
def home():
    return render_template('upload.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('upload.html', result="No file part")
    file = request.files['file']
    if file.filename == '':
        return render_template('upload.html', result="No selected file")
    try:
        img_bytes = file.read()
        img_array = preprocess_image(img_bytes)
        prediction = model.predict(img_array)
        pred_label = int(prediction[0][0] > 0.5)
        if pred_label == 0:
            result = "Image is AI Generated"
        else:
            result = "Image is Real"
        return render_template('upload.html', result=result)
    except Exception as e:
        return render_template('upload.html', result=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
