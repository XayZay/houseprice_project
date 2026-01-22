"""
House Price Prediction Model Development
This script develops a machine learning model to predict house prices
using the House Prices: Advanced Regression Techniques dataset.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

def load_dataset():
    """Load the house prices dataset"""
    print("Loading dataset...")
    try:
        # Try to load from local path first
        train_data = pd.read_csv('train.csv')
        print(f"Dataset loaded successfully. Shape: {train_data.shape}")
        return train_data
    except FileNotFoundError:
        print("ERROR: Dataset not found!")
        print("Please download train.csv from:")
        print("https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data")
        raise

def preprocess_data(df):
    """
    Perform data preprocessing:
    - Handle missing values
    - Feature selection (6 features from recommended 9)
    - Encoding categorical variables
    - Feature scaling
    """
    print("\n" + "="*50)
    print("PREPROCESSING DATA")
    print("="*50)
    
    # Create a copy to avoid modifying original
    data = df.copy()
    
    # Map actual dataset columns to expected feature names
    # Mapping for House Prices dataset (if columns match Kaggle dataset)
    kaggle_features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'YearBuilt', 'Neighborhood']
    kaggle_target = 'SalePrice'
    
    # Alternative mapping for other real estate datasets
    alternative_features = ['condition', 'sqft_living', 'sqft_basement', 'bedrooms', 'yr_built', 'city']
    alternative_target = 'price'
    
    # Detect which dataset we have
    kaggle_exists = all(f in data.columns for f in kaggle_features + [kaggle_target])
    alt_exists = all(f in data.columns for f in alternative_features + [alternative_target])
    
    if kaggle_exists:
        selected_features = kaggle_features
        target = kaggle_target
        print("Detected: House Prices Kaggle Dataset")
    elif alt_exists:
        selected_features = alternative_features
        target = alternative_target
        print("Detected: Alternative Real Estate Dataset")
        # Rename columns to match expected names
        data = data.rename(columns={
            'condition': 'OverallQual',
            'sqft_living': 'GrLivArea',
            'sqft_basement': 'TotalBsmtSF',
            'bedrooms': 'GarageCars',
            'yr_built': 'YearBuilt',
            'city': 'Neighborhood',
            'price': 'SalePrice'
        })
        selected_features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'YearBuilt', 'Neighborhood']
        target = 'SalePrice'
    else:
        raise ValueError(f"Dataset columns do not match expected features.\nDataset columns: {list(data.columns)}\nExpected (Kaggle): {kaggle_features}\nExpected (Alternative): {alternative_features}")
    
    # Select only the features we need
    data = data[selected_features + [target]].copy()
    
    print(f"\nSelected Features (6 of 9):")
    for i, feature in enumerate(selected_features, 1):
        print(f"  {i}. {feature}")
    
    print(f"\nTarget: {target}")
    
    print(f"\nMissing values BEFORE handling:")
    missing = data.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  No missing values")
    
    # Handle missing values
    numerical_features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'YearBuilt']
    for feature in numerical_features:
        if data[feature].isnull().sum() > 0:
            median_val = data[feature].median()
            print(f"  Filling {feature} missing values with median: {median_val}")
            data[feature].fillna(median_val, inplace=True)
    
    # For categorical feature (Neighborhood): fill with mode
    categorical_feature = 'Neighborhood'
    if data[categorical_feature].isnull().sum() > 0:
        mode_val = data[categorical_feature].mode()[0]
        print(f"  Filling {categorical_feature} missing values with mode: {mode_val}")
        data[categorical_feature].fillna(mode_val, inplace=True)
    
    # Target variable: remove rows with missing values
    initial_rows = len(data)
    data = data.dropna(subset=[target])
    removed_rows = initial_rows - len(data)
    if removed_rows > 0:
        print(f"  Removed {removed_rows} rows with missing target values")
    
    print(f"\nMissing values AFTER handling:")
    missing = data.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  No missing values")
    
    # Separate features and target
    X = data[selected_features].copy()
    y = data[target].copy()
    
    # Encode categorical variables
    print("\nEncoding categorical variables...")
    le_dict = {}
    categorical_features = ['Neighborhood']
    
    for feature in categorical_features:
        le = LabelEncoder()
        X[feature] = le.fit_transform(X[feature])
        le_dict[feature] = le
        print(f"  {feature}: {len(le.classes_)} unique categories encoded")
    
    # Save label encoders for later use
    os.makedirs('models', exist_ok=True)
    joblib.dump(le_dict, 'models/label_encoders.joblib')
    
    # Feature scaling
    print("\nApplying feature scaling (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=selected_features)
    
    # Save scaler for later use
    joblib.dump(scaler, 'models/scaler.joblib')
    
    print(f"\nFinal dataset shape:")
    print(f"  Features (X): {X.shape}")
    print(f"  Target (y): {y.shape}")
    
    return X, y, selected_features

def train_model(X, y):
    """
    Train Random Forest Regressor model
    """
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    
    # Split data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nDataset split (80-20 train-test):")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Testing samples: {X_test.shape[0]}")
    
    # Create and train the model
    print(f"\nAlgorithm: Random Forest Regressor")
    print(f"  Number of trees: 100")
    print(f"  Max depth: 20")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print(f"\nModel training completed!")
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using regression metrics
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nRegression Metrics:")
    print(f"  Mean Absolute Error (MAE):        ${mae:,.2f}")
    print(f"  Mean Squared Error (MSE):         ${mse:,.2f}")
    print(f"  Root Mean Squared Error (RMSE):   ${rmse:,.2f}")
    print(f"  R² Score:                         {r2:.4f}")
    
    # Feature importance
    features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'YearBuilt', 'Neighborhood']
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\nFeature Importance Ranking:")
    for idx, row in feature_importance.iterrows():
        print(f"  {row['Feature']:15s}: {row['Importance']:.4f}")
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

def save_model(model, metrics):
    """
    Save the trained model and metrics to disk
    """
    print("\n" + "="*50)
    print("SAVING MODEL")
    print("="*50)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    joblib.dump(model, 'models/price_model.joblib')
    print(f"\n✓ Model saved: models/price_model.joblib")
    
    # Save metrics
    joblib.dump(metrics, 'models/model_metrics.joblib')
    print(f"✓ Metrics saved: models/model_metrics.joblib")
    
    # Save feature names for later use
    feature_names = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'YearBuilt', 'Neighborhood']
    joblib.dump(feature_names, 'models/feature_names.joblib')
    print(f"✓ Feature names saved: models/feature_names.joblib")

def main():
    """Main execution function"""
    print("\n")
    print("╔" + "="*48 + "╗")
    print("║" + " "*48 + "║")
    print("║" + "  HOUSE PRICE PREDICTION MODEL DEVELOPMENT".center(48) + "║")
    print("║" + " "*48 + "║")
    print("╚" + "="*48 + "╝")
    
    try:
        # Load dataset
        df = load_dataset()
        
        # Preprocess data
        X, y, features = preprocess_data(df)
        
        # Train model
        model, X_test, y_test = train_model(X, y)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save model
        save_model(model, metrics)
        
        print("\n" + "="*50)
        print("✓ MODEL DEVELOPMENT COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("\nYou can now use the trained model in the Flask app.")
        print("Run: python app.py")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()
