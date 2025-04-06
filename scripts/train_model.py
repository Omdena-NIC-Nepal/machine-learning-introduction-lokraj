import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
import os
import sys

def validate_processed_data(X_train: pd.DataFrame) -> None:
    """Verify standardization was applied correctly"""
    if not (X_train.mean().abs() < 1e-2).all():
        raise ValueError("Features not properly centered (mean≠0)")
    if not (X_train.std().round(2) == 1.0).all():
        print("Warning: Some features not perfectly scaled (std≠1)", file=sys.stderr)

def train_model(X_path: str, y_path: str, features: list, output_path: str) -> None:
    """Train and save a linear regression model with standardization"""
    try:
        # 1. Load data
        X_train = pd.read_csv(X_path)
        y_train = pd.read_csv(y_path).squeeze()
        
        # 2. Feature selection
        missing = [f for f in features if f not in X_train.columns]
        if missing:
            raise KeyError(f"Features not found: {missing}")
        X_train = X_train[features]
        
        # 3. Standardize features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        
        # 4. Validate standardization
        validate_processed_data(X_train_scaled)
        
        # 5. Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # 6. Save artifacts
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump({
            'model': model,
            'scaler': scaler,
            'features': features,
            'preprocessing': 'standard_scaler'
        }, output_path)
        
        # 7. Training metrics
        y_pred = model.predict(X_train_scaled)
        print(f"\nTraining Metrics:")
        print(f"- MSE: {mean_squared_error(y_train, y_pred):.2f}")
        print(f"- R2: {r2_score(y_train, y_pred):.2f}")
        print(f"\nModel saved to {output_path}")
        print(f"Features used: {features}")

    except Exception as e:
        print(f"\nError in model training: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    DEFAULT_FEATURES = ['rm', 'lstat', 'ptratio', 'indus', 'nox']
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--X", 
                       default=os.path.join("..", "data", "processed", "X_train.csv"),
                       help="Path to training features")
    parser.add_argument("--y", 
                       default=os.path.join("..", "data", "processed", "y_train.csv"),
                       help="Path to training target")
    parser.add_argument("--features", 
                       nargs='+', 
                       default=DEFAULT_FEATURES,
                       help="List of features to use")
    parser.add_argument("--output", 
                       default=os.path.join("..", "models", "housing_model.pkl"),
                       help="Output path for trained model")
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.X = os.path.abspath(os.path.join(script_dir, args.X))
    args.y = os.path.abspath(os.path.join(script_dir, args.y))
    args.output = os.path.abspath(os.path.join(script_dir, args.output))
    
    print(f"Training model with:")
    print(f"- Features: {args.X}")
    print(f"- Target: {args.y}")
    print(f"- Selected features: {args.features}")
    
    train_model(args.X, args.y, args.features, args.output)