import numpy as np
from flask import Flask, request, render_template
import joblib
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
model = None
model_path = os.path.join(os.path.dirname(__file__), 'Diabetes.pkl')

try:
    loaded_object = joblib.load(model_path)
    if callable(loaded_object):
        # If it's a function, it might be a pipeline or a custom object
        model = loaded_object
    elif hasattr(loaded_object, 'predict'):
        # If it has a predict method, it's likely a sklearn model
        model = loaded_object
    else:
        raise TypeError("Loaded object is neither callable nor has a 'predict' method")
    logging.info("Model loaded successfully.")
except FileNotFoundError:
    logging.error(f"Model file not found at path: {model_path}")
except Exception as e:
    logging.error(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            raise ValueError("Model not loaded")

        # Extract features from the form
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)
        logging.info(f"Features received: {features}")
        logging.info(f"Final features for prediction: {final_features}")

        # Make a prediction
        if callable(model):
            prediction = model(final_features)[0]
        else:
            prediction = model.predict(final_features)[0]
        logging.info(f"Prediction result: {prediction}")

        # Determine prediction text
        prediction_text = 'High chances of patient having diabetes' if prediction == 1 else 'Low chances of patient having diabetes'

        return render_template('index.html', prediction_text=prediction_text)
    except Exception as e:
        error_message = f"Error during prediction: {str(e)}"
        logging.error(error_message)
        return render_template('index.html', prediction_text=error_message)

if __name__ == "__main__":
    app.run(debug=True)
