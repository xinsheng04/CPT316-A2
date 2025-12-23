from flask import Flask, request, jsonify
import numpy as np
from tensorflow import keras

app = Flask(__name__)
r'''
-----------------------------------------------
Paste the following into PowerShell before running node-red:
cd 'C:\Users\Asus\OneDrive\Desktop\myProjects\Y3S1\CPT 316\a2'
-----------------------------------------------
'''
# Load model once - stays in RAM
print("---Keras Loads Model---")
model = keras.models.load_model("digit_classifier.keras")
print("---Model Loaded---")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        pixels = np.array(data, dtype=np.float32).reshape(1, 784)
        
        preds = model.predict(pixels, verbose=0)

        return jsonify({
            "predictions": preds.tolist()
        })
    except Exception as e:
        print(f"Python Error: {e}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(port=5000, debug=True)