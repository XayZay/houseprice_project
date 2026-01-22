"""
House Price Prediction Web Application
Flask application for house price prediction using trained ML model
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import traceback

app = Flask(__name__)

# Global variables for model and preprocessing tools
model = None
scaler = None
label_encoders = None
feature_names = None
metrics = None

def load_model_artifacts():
    """Load the trained model and preprocessing artifacts"""
    global model, scaler, label_encoders, feature_names, metrics
    
    try:
        # Check if model files exist
        model_path = 'models/price_model.joblib'
        scaler_path = 'models/scaler.joblib'
        le_path = 'models/label_encoders.joblib'
        features_path = 'models/feature_names.joblib'
        metrics_path = 'models/model_metrics.joblib'
        
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}")
            print("Please run: python model_development.py")
            return False
        
        print("Loading model artifacts...")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoders = joblib.load(le_path)
        feature_names = joblib.load(features_path)
        metrics = joblib.load(metrics_path)
        
        print("✓ Model loaded successfully")
        return True
        
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """Render the main page"""
    if model is None:
        return "Error: Model not loaded. Please run model_development.py first.", 500
    return render_template('index.html', metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a prediction based on user input
    """
    try:
        data = request.get_json()
        
        # Extract features from request
        overall_qual = float(data.get('overall_qual'))
        gr_liv_area = float(data.get('gr_liv_area'))
        total_bsmt_sf = float(data.get('total_bsmt_sf'))
        garage_cars = float(data.get('garage_cars'))
        year_built = float(data.get('year_built'))
        neighborhood = data.get('neighborhood')
        
        # Validate inputs
        if not all([overall_qual, gr_liv_area, total_bsmt_sf, garage_cars, year_built, neighborhood]):
            return jsonify({'error': 'All fields are required'}), 400
        
        # Validate ranges
        if overall_qual < 1 or overall_qual > 10:
            return jsonify({'error': 'Overall Quality must be between 1 and 10'}), 400
        if gr_liv_area < 0:
            return jsonify({'error': 'GrLivArea must be positive'}), 400
        if total_bsmt_sf < 0:
            return jsonify({'error': 'TotalBsmtSF must be positive'}), 400
        if garage_cars < 0 or garage_cars > 5:
            return jsonify({'error': 'Garage Cars must be between 0 and 5'}), 400
        if year_built < 1800 or year_built > 2024:
            return jsonify({'error': 'Year Built must be between 1800 and 2024'}), 400
        
        # Prepare features for prediction
        features = np.array([[
            overall_qual,
            gr_liv_area,
            total_bsmt_sf,
            garage_cars,
            year_built,
            label_encoders['Neighborhood'].transform([neighborhood])[0]
        ]])
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Ensure prediction is positive
        prediction = max(0, prediction)
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'formatted_price': f"${prediction:,.2f}"
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/neighborhoods', methods=['GET'])
def get_neighborhoods():
    """Return list of available neighborhoods"""
    try:
        neighborhoods = list(label_encoders['Neighborhood'].classes_)
        neighborhoods.sort()
        return jsonify({'neighborhoods': neighborhoods})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Return model metrics"""
    try:
        return jsonify({
            'MAE': float(metrics['MAE']),
            'RMSE': float(metrics['RMSE']),
            'R2': float(metrics['R2'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("HOUSE PRICE PREDICTION WEB APPLICATION")
    print("="*50)
    
    # Load model artifacts
    if load_model_artifacts():
        print("\nStarting Flask server...")
        print("Open your browser and go to: http://localhost:5000")
        print("\nPress Ctrl+C to stop the server")
        app.run(debug=True, host='127.0.0.1', port=5000)
    else:
        print("\n✗ Failed to load model. Please ensure model_development.py has been run.")
