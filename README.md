# Handwritten Digit Classifier

A machine learning application that recognizes handwritten digits (0-9) using a Convolutional Neural Network built with Keras and trained on the MNIST handwritten digits dataset. Contains an interactive web interface powered by Node-RED allowing users to draw digits and receive real-time predictions.

## Project Structure

- `HandwrittenDigitClassifier.ipynb` - Jupyter notebook for model training and evaluation
- `predict.py` - Flask server providing the prediction API
- `digit_classifier_CNN.weights.h5` - Pre-trained model weights
- `Node-RED/` - Node-RED flow configuration and dashboard assets

## Features

- Interactive drawing canvas for digit input
- Real-time digit prediction
- Pre-trained CNN model for accurate recognition
- User-friendly Node-RED dashboard interface

## Tech Stack

- Frontend: Node-RED Dashboard
- Backend: Python Flask
- Machine Learning: TensorFlow, Keras
- Data Processing: Scikit-learn
- Model: Convolutional Neural Network (CNN)

## Dependencies

### Python Dependencies
- `flask` - Web framework for the prediction API
- `tensorflow` - Machine learning framework
- `keras` - For initializing and training neural networks
- `numpy` - Numerical computing library
- `scipy` - Image processing library

### Node.js Dependencies
- `node-red` - Flow-based programming tool for the dashboard interface

### Setup and Installation

## Dependencies installation

To install Python dependencies, run:
```bash
pip install flask tensorflow numpy scipy
```

To install Node-RED, run:
```bash
npm install -g node-red
```

### Starting Node-RED

1. Get the root directory of this project, for example, C:\Users\Asus\OneDrive\Desktop\myProjects\Y3S1\CPT 316\a2
2. In your PowerShell command prompt (Windows Powershell) paste 
```bash 
cd [your-root-directory]
```
 to switch to that root directory
- For example, 
```bash
cd 'C:\Users\Asus\OneDrive\Desktop\myProjects\Y3S1\CPT 316\a2'
```
3. In that very same directory, run node-red and open Node-red in the browser

### Starting the Predictor Function

4. Open another command prompt (Windows PowerShell) in the same root directory.
5. run the following command: 
```bash
python predict.py
```
6. Wait for the words " * Debugger is active! * Debugger PIN: 361-710-501" to appear in the command prompt

## Usage

Once both Node-RED and the prediction service are running, navigate to the Node-RED dashboard in your browser. Use the drawing canvas to write a digit, then submit it for prediction. The model will analyze your input and display the predicted digit.

## Important Notes

- Digit prediction accuracy depends on the quality of the drawn input and the model's training
- The application performs best with clearly drawn, centered digits
- Prediction is not 100% accurate