from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('svr_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize the Flask app
app = Flask(__name__)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the POST request
    data = request.get_json()
    
    # Extract features (expecting the input to be a dictionary with 'features' key)
    features = np.array(data['features']).reshape(1, -1)  # Reshaping as needed for the model input
    
    # Scale the features using the loaded scaler
    features_scaled = scaler.transform(features)
    
    # Predict using the trained model
    prediction = model.predict(features_scaled)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction[0]})

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)