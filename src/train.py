"""
Sales Prediction Training Script
Author: Harsh (Junior Developer)
Date: July 2026
Purpose: Performs exploratory data analysis, trains multiple regression models,
         evaluates performance, saves the best model, and generates analysis plots.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set plotting style for premium aesthetics
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'figure.facecolor': '#121212',
    'axes.facecolor': '#1e1e1e',
    'text.color': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#b0b0b0',
    'ytick.color': '#b0b0b0',
    'axes.titlecolor': '#ffffff',
    'grid.color': '#333333',
    'font.family': 'sans-serif',
})

# Define paths
DATA_PATH = os.path.join("sales_prediction", "data", "Advertising.csv")
PLOTS_DIR = os.path.join("sales_prediction", "plots")
MODELS_DIR = os.path.join("sales_prediction", "models")

# Ensure directories exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data():
    """Load the advertising dataset, dropping the index column."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    
    # Read CSV, setting the first column as index (since it is an ID column)
    df = pd.read_csv(DATA_PATH, index_col=0)
    print("Dataset Loaded Successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df

def perform_eda(df):
    """Generate and save EDA plots."""
    print("\n--- Performing Exploratory Data Analysis ---")
    
    # 1. Correlation Matrix
    plt.figure(figsize=(8, 6))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, mask=mask, cmap="coolwarm", fmt=".3f", 
                linewidths=1, cbar_kws={"shrink": .8}, annot_kws={"size": 12})
    plt.title("Correlation Heatmap of Advertising Spend & Sales", fontsize=14, pad=15)
    plt.tight_layout()
    corr_path = os.path.join(PLOTS_DIR, "correlation_matrix.png")
    plt.savefig(corr_path, dpi=300, facecolor='#121212')
    plt.close()
    print(f"Saved: {corr_path}")

    # 2. Pairwise Spend vs Sales Regression Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    features = ['TV', 'Radio', 'Newspaper']
    colors = ['#ff79c6', '#8be9fd', '#50fa7b']
    
    for i, feature in enumerate(features):
        sns.regplot(
            data=df, x=feature, y='Sales', ax=axes[i],
            scatter_kws={'alpha': 0.6, 'color': colors[i], 'edgecolor': 'none', 's': 40},
            line_kws={'color': '#ffb86c', 'linewidth': 2}
        )
        axes[i].set_title(f"{feature} Spend vs Sales", fontsize=14, pad=10)
        axes[i].set_xlabel(f"{feature} Advertising Budget ($)", fontsize=12)
        if i == 0:
            axes[i].set_ylabel("Sales (units)", fontsize=12)
            
    plt.suptitle("Impact of Advertising Channels on Sales Outcomes", fontsize=16, y=1.02, color='#ffffff')
    plt.tight_layout()
    spend_path = os.path.join(PLOTS_DIR, "spend_vs_sales.png")
    plt.savefig(spend_path, dpi=300, facecolor='#121212')
    plt.close()
    print(f"Saved: {spend_path}")

    # Calculate VIF to check multicollinearity
    print("\nVariance Inflation Factors (VIF) check:")
    X_vif = df[['TV', 'Radio', 'Newspaper']]
    for col in X_vif.columns:
        X_other = X_vif.drop(columns=[col])
        y_col = X_vif[col]
        lr = LinearRegression()
        lr.fit(X_other, y_col)
        r2 = lr.score(X_other, y_col)
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float('inf')
        print(f"  {col}: {vif:.4f}")


def train_and_evaluate_models(df):
    """Train multiple regression models and find the best one."""
    print("\n--- Model Development & Evaluation ---")
    
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features (recommended for linear models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Linear Regression': (LinearRegression(), True),
        'Ridge Regression': (Ridge(alpha=1.0), True),
        'Lasso Regression': (Lasso(alpha=0.1), True),
        'Random Forest': (RandomForestRegressor(n_estimators=100, random_state=42), False),
        'Gradient Boosting': (GradientBoostingRegressor(n_estimators=100, random_state=42), False)
    }
    
    results = {}
    best_model_name = None
    best_r2 = -float('inf')
    best_model_obj = None
    best_model_needs_scaling = False
    
    for name, (model, needs_scaling) in models.items():
        X_tr = X_train_scaled if needs_scaling else X_train
        X_te = X_test_scaled if needs_scaling else X_test
        
        # Train model
        model.fit(X_tr, y_train)
        
        # Predict
        preds = model.predict(X_te)
        
        # Metrics
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        
        # Cross-validation on full dataset
        X_cv = scaler.fit_transform(X) if needs_scaling else X
        cv_scores = cross_val_score(model, X_cv, y, cv=5, scoring='r2')
        cv_mean = np.mean(cv_scores)
        
        results[name] = {
            'R2': r2,
            'MAE': mae,
            'RMSE': rmse,
            'CV_R2': cv_mean,
            'model_obj': model,
            'needs_scaling': needs_scaling
        }
        
        print(f"\nModel: {name}")
        print(f"  R2 Score: {r2:.4f}")
        print(f"  MAE:      {mae:.4f}")
        print(f"  RMSE:     {rmse:.4f}")
        print(f"  5-Fold CV R2 Mean: {cv_mean:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model_obj = model
            best_model_needs_scaling = needs_scaling
            
    print(f"\n>>> Best Model: {best_model_name} (R2: {best_r2:.4f})")
    
    # Save model and preprocessing scaler if needed
    model_save_dict = {
        'model': best_model_obj,
        'model_name': best_model_name,
        'scaler': scaler if best_model_needs_scaling else None,
        'features': list(X.columns),
        'metrics': {
            'R2': results[best_model_name]['R2'],
            'MAE': results[best_model_name]['MAE'],
            'RMSE': results[best_model_name]['RMSE']
        }
    }
    
    model_pkl_path = os.path.join(MODELS_DIR, "sales_model.pkl")
    with open(model_pkl_path, 'wb') as f:
        pickle.dump(model_save_dict, f)
    print(f"Best model saved to {model_pkl_path}")
    
    # Plot model comparison
    plot_model_comparison(results)
    
    # Plot best model performance diagnostics
    X_te_best = X_test_scaled if best_model_needs_scaling else X_test
    best_preds = best_model_obj.predict(X_te_best)
    plot_model_diagnostics(y_test, best_preds, best_model_name, best_model_obj, X.columns)
    
    return results, best_model_name

def plot_model_comparison(results):
    """Plot bar charts comparing the performance of different models."""
    names = list(results.keys())
    r2_scores = [results[n]['R2'] for n in names]
    rmse_scores = [results[n]['RMSE'] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # R2 bars
    color = '#ff79c6'
    ax1.set_xlabel('Regression Models', fontsize=12, labelpad=10)
    ax1.set_ylabel('R-squared Score', color=color, fontsize=12)
    rects1 = ax1.bar(x - width/2, r2_scores, width, label='R-squared', color=color, alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0.8, 1.0)
    
    # RMSE bars (on secondary axis)
    ax2 = ax1.twinx()
    color = '#8be9fd'
    ax2.set_ylabel('RMSE (Sales Units)', color=color, fontsize=12)
    rects2 = ax2.bar(x + width/2, rmse_scores, width, label='RMSE', color=color, alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha='right')
    plt.title("Model Comparison: R-squared and RMSE Scores", fontsize=14, pad=15)
    fig.tight_layout()
    
    comp_path = os.path.join(PLOTS_DIR, "model_comparison.png")
    plt.savefig(comp_path, dpi=300, facecolor='#121212')
    plt.close()
    print(f"Saved: {comp_path}")

def plot_model_diagnostics(y_true, y_pred, model_name, model_obj, feature_names):
    """Plot diagnostic charts for the best performing model."""
    residuals = y_true - y_pred
    
    # 1. Predictions vs Actual Scatter Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, color='#50fa7b', alpha=0.8, s=60, edgecolor='none')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(f"{model_name}: Predicted vs Actual Sales", fontsize=14, pad=15)
    plt.xlabel("Actual Sales", fontsize=12)
    plt.ylabel("Predicted Sales", fontsize=12)
    plt.tight_layout()
    diag_path1 = os.path.join(PLOTS_DIR, "actual_vs_predicted.png")
    plt.savefig(diag_path1, dpi=300, facecolor='#121212')
    plt.close()
    print(f"Saved: {diag_path1}")
    
    # 2. Residuals Distribution Plot
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='#bd93f9', bins=15, edgecolor='#121212')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=1.5)
    plt.title(f"{model_name}: Distribution of Residuals (Errors)", fontsize=14, pad=15)
    plt.xlabel("Residual (Actual - Predicted)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    diag_path2 = os.path.join(PLOTS_DIR, "residuals_distribution.png")
    plt.savefig(diag_path2, dpi=300, facecolor='#121212')
    plt.close()
    print(f"Saved: {diag_path2}")
    
    # 3. Feature Importance or Coefficients Plot
    plt.figure(figsize=(8, 6))
    if hasattr(model_obj, 'feature_importances_'):
        importances = model_obj.feature_importances_
        title = f"{model_name}: Feature Importances"
        ylabel = "Relative Importance"
        values = importances
    elif hasattr(model_obj, 'coef_'):
        title = f"{model_name}: Regression Coefficients"
        ylabel = "Coefficient Value"
        values = model_obj.coef_
    else:
        return
        
    sns.barplot(x=list(feature_names), y=values, palette=['#ff79c6', '#8be9fd', '#50fa7b'], hue=list(feature_names), legend=False)
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Advertising Channel", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    diag_path3 = os.path.join(PLOTS_DIR, "advertising_impact.png")
    plt.savefig(diag_path3, dpi=300, facecolor='#121212')
    plt.close()
    print(f"Saved: {diag_path3}")

if __name__ == "__main__":
    df = load_data()
    
    # Check nulls and descriptive statistics
    print("\nDescriptive Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    perform_eda(df)
    results, best_model = train_and_evaluate_models(df)
    print("\n--- Training Pipeline Complete ---")
