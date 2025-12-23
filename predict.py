from flask import Flask, request, jsonify
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from scipy import ndimage

app = Flask(__name__)
r'''
-----------------------------------------------
Paste the following into PowerShell before running node-red:
cd 'C:\Users\Asus\OneDrive\Desktop\myProjects\Y3S1\CPT 316\a2'
-----------------------------------------------
'''
# Load model once - stays in RAM
print("---Keras Loads Model---")

num_filters = 8
filter_size = 3
pool_size = 2
model = Sequential()
model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.load_weights('digit_classifier_CNN.weights.h5')
print("---Model Loaded---")

def center_image(img):
    cy, cx = ndimage.center_of_mass(img)
    if np.isnan(cx) or np.isnan(cy):
        return img
    shift_x = int(np.round(14 - cx))
    shift_y = int(np.round(14 - cy))
    return ndimage.shift(img, shift=(shift_y, shift_x), mode='constant')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Reshape the image to 28x28
        pixels = np.array(data, dtype=np.float32).reshape(28, 28)
        
        # Center the image
        img = center_image(pixels)
        
        # Clip pixel values to [0, 1]
        img = np.clip(img, 0.0, 1.0)
        
        # Subtract values to normalize around 0 [-0.5, 0.5]
        img = img - 0.5
        
        # Reshape for model input
        img = img.reshape(1, 28, 28, 1)
        
        # Make prediction and return predictions for Node-red interpretation
        preds = model.predict(img, verbose=0)

        return jsonify({
            "predictions": preds.tolist()
        })
    except Exception as e:
        print(f"Python Error: {e}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(port=5000, debug=True)