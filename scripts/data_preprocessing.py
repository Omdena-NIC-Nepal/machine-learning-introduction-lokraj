import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse

def preprocess_data(input_path, output_dir):
    """Handle all preprocessing steps"""
    # Load data
    df = pd.read_csv(input_path)
    
    # Outlier handling
    df['crim'] = df['crim'].clip(upper=df['crim'].quantile(0.99))
    df['log_crim'] = np.log1p(df['crim'])
    df['log_lstat'] = np.log1p(df['lstat'])
    
    # Feature scaling
    numeric_features = ['crim', 'zn', 'indus', 'nox', 'rm', 
                      'age', 'dis', 'rad', 'tax', 'ptratio',
                      'b', 'lstat', 'log_crim', 'log_lstat']
    
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    # Train-test split
    X = df.drop('medv', axis=1)
    y = df['medv']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save processed data
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    
    print(f"Processed data saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./data/boston_housing.csv",
                       help="Input raw data path")
    parser.add_argument("--output", default="./data/processed",
                       help="Output directory for processed data")
    args = parser.parse_args()
    
    preprocess_data(args.input, args.output)