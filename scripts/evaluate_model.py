import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import argparse
import os
import sys

def load_data(X_path: str, y_path: str, features: list) -> tuple:
    """Load and prepare test data."""
    X_test = pd.read_csv(X_path)
    y_test = pd.read_csv(y_path).squeeze()
    
    # Select specified features
    missing = [f for f in features if f not in X_test.columns]
    if missing:
        raise KeyError(f"Features not found: {missing}")
    return X_test[features], y_test

def evaluate_model(model_path: str, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate model performance and generate plots."""
    artifacts = joblib.load(model_path)
    model = artifacts['model']
    scaler = artifacts['scaler']
    
    # Preprocess test data 
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    metrics = {
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    
    # Residuals plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.savefig('residuals_plot.png')
    plt.close()
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", 
                       default=os.path.join("..", "models", "housing_model.pkl"),
                       help="Path to trained model")
    parser.add_argument("--X_test", 
                       default=os.path.join("..", "data", "processed", "X_test.csv"),
                       help="Path to test features")
    parser.add_argument("--y_test", 
                       default=os.path.join("..", "data", "processed", "y_test.csv"),
                       help="Path to test target")
    parser.add_argument("--features",
                       nargs='+',
                       default=['rm', 'lstat', 'ptratio', 'indus', 'nox'],
                       help="Features used in training")
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.model = os.path.abspath(os.path.join(script_dir, args.model))
    args.X_test = os.path.abspath(os.path.join(script_dir, args.X_test))
    args.y_test = os.path.abspath(os.path.join(script_dir, args.y_test))
    
    try:
        X_test, y_test = load_data(args.X_test, args.y_test, args.features)
        metrics = evaluate_model(args.model, X_test, y_test)
        
        print("\nEvaluation Metrics:")
        for name, value in metrics.items():
            print(f"- {name}: {value:.4f}")
        print(f"\nResiduals plot saved to residuals_plot.png")
        
    except Exception as e:
        print(f"Error in evaluation: {str(e)}", file=sys.stderr)
        sys.exit(1)