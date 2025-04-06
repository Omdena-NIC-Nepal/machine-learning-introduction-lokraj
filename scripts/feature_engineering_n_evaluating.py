"""
feature_engineering.py - Complete implementation of Boston Housing feature engineering pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
import os

def load_data():
    """Load and prepare raw data"""
    df = pd.read_csv("./data/boston_housing.csv")
    print(f"Original dataset shape: {df.shape}")
    
    # Standardize column names
    df.columns = df.columns.str.lower()
    return df

def create_features(df):
    """
    Generate new features for model improvement
    Returns DataFrame with original + engineered features
    """
    df = df.copy()
    
    # Interaction terms
    df['rooms_per_tax'] = df['rm'] / (df['tax'] + 1e-6)
    df['nox_access'] = df['nox'] * df['dis']
    
    # Polynomial transforms
    df['lstat_sq'] = df['lstat'] ** 2
    df['crim_log'] = np.log1p(df['crim'])
    
    # Proximity metrics
    df['highway_access'] = df['rad'] / (df['dis'] + 1)
    
    # Binning continuous variables
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 50, 80, 100], 
                            labels=['new','mid','old'])
    
    return df

def visualize_features(df, features, target='medv'):
    """Plot relationships between new features and target"""
    plt.figure(figsize=(12, 8))
    for i, feat in enumerate(features[:4], 1):
        plt.subplot(2, 2, i)
        sns.scatterplot(x=feat, y=target, data=df, alpha=0.6)
        plt.title(f"{feat} vs {target}")
    plt.tight_layout()
    plt.savefig("./reports/feature_relationships.png")
    plt.close()

def select_features(X, y, n_features=10):
    """Perform feature selection using RFE"""
    selector = RFE(LinearRegression(), n_features_to_select=n_features)
    selector.fit(X, y)
    return X.columns[selector.support_]

def evaluate_models(X_train, X_test, y_train, y_test, baseline_features):
    """Compare baseline vs enhanced models"""
    # Baseline model
    baseline = LinearRegression()
    baseline.fit(X_train[baseline_features], y_train)
    baseline_pred = baseline.predict(X_test[baseline_features])
    
    # Enhanced model
    enhanced = LinearRegression()
    enhanced.fit(X_train, y_train)
    enhanced_pred = enhanced.predict(X_test)
    
    # Compare results
    results = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'R2'],
        'Baseline': [
            mean_squared_error(y_test, baseline_pred),
            np.sqrt(mean_squared_error(y_test, baseline_pred)),
            r2_score(y_test, baseline_pred)
        ],
        'Enhanced': [
            mean_squared_error(y_test, enhanced_pred),
            np.sqrt(mean_squared_error(y_test, enhanced_pred)),
            r2_score(y_test, enhanced_pred)
        ]
    })
    
    results['Improvement%'] = ((results['Enhanced'] - results['Baseline']) / 
                              results['Baseline'] * 100).round(1)
    
    return results, enhanced

def save_results(engineered_df, selected_features, model, output_dir="../models"):
    """Save processed data and model artifacts"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    X_processed = pd.get_dummies(engineered_df)[selected_features]
    X_processed.to_csv("./data/processed/X_engineered_reg.csv", index=False)
    engineered_df['medv'].to_csv("./data/processed/y_engineered_reg.csv", index=False)
    
    # Save feature list
    with open(os.path.join(output_dir, "selected_features.txt"), "w") as f:
        f.write("\n".join(selected_features))
    
    # Save model
    joblib.dump(model, os.path.join(output_dir, "enhanced_model.pkl"))

def main():
    # 1. Load and prepare data
    df = load_data()
    
    # 2. Feature engineering
    engineered_df = create_features(df)
    new_features = list(set(engineered_df.columns) - set(df.columns))
    print(f"\nCreated {len(new_features)} new features: {new_features}")
    
    # 3. Visualize new features
    visualize_features(engineered_df, new_features)
    
    # 4. Prepare data for modeling
    X = pd.get_dummies(engineered_df.drop('medv', axis=1))
    y = engineered_df['medv']
    
    # 5. Feature selection
    selected_features = select_features(X, y)
    print("\nSelected features:", selected_features.tolist())
    
    # 6. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X[selected_features], y, test_size=0.2, random_state=42
    )
    
    # 7. Model evaluation
    baseline_features = ['crim_log', 'rm', 'lstat', 'ptratio', 'nox']
    results, enhanced_model = evaluate_models(
        X_train, X_test, y_train, y_test, baseline_features
    )
    
    # 8. Save results
    save_results(engineered_df, selected_features, enhanced_model)
    
    # 9. Display comparison
    print("\n=== Performance Comparison ===")
    print(results)
    print("\nFeature relationships plot saved to ./reports/feature_relationships.png")
    print("Processed data and model saved to ../data/processed/ and ../models/")

if __name__ == "__main__":
    main()