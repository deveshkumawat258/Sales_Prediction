"""
Sales Prediction Budget Optimizer
Author: Harsh (Junior Developer)
Date: July 2026
Purpose: Provides functions to optimize advertising budget allocation across
         TV, Radio, and Newspaper channels to maximize predicted sales using SciPy.
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Paths
MODELS_DIR = os.path.join("sales_prediction", "models")
MODEL_PKL_PATH = os.path.join(MODELS_DIR, "sales_model.pkl")
DATA_PATH = os.path.join("sales_prediction", "data", "Advertising.csv")

class BudgetOptimizer:
    def __init__(self):
        self.model_loaded = False
        self.best_model = None
        self.scaler = None
        self.features = []
        self.lr_model = None
        self.lr_scaler = None
        
        # Load the best model and train a linear model for comparison
        self._load_models()

    def _load_models(self):
        """Load the saved best model and train a companion linear model."""
        # 1. Load Saved Best Model (Gradient Boosting)
        if os.path.exists(MODEL_PKL_PATH):
            try:
                with open(MODEL_PKL_PATH, 'rb') as f:
                    saved_data = pickle.load(f)
                self.best_model = saved_data['model']
                self.scaler = saved_data['scaler']
                self.features = saved_data['features']
                self.model_loaded = True
                print(f"Loaded best model: {saved_data['model_name']}")
            except Exception as e:
                print(f"Error loading saved model: {e}")
        
        # 2. Train a Linear Regression model on the fly for linear optimization comparison
        if os.path.exists(DATA_PATH):
            try:
                df = pd.read_csv(DATA_PATH, index_col=0)
                X = df[['TV', 'Radio', 'Newspaper']]
                y = df['Sales']
                
                self.lr_scaler = StandardScaler()
                X_scaled = self.lr_scaler.fit_transform(X)
                
                self.lr_model = LinearRegression()
                self.lr_model.fit(X_scaled, y)
                print("Trained Linear Regression companion model for smooth SLSQP optimization.")
            except Exception as e:
                print(f"Error training companion linear model: {e}")

    def _predict_best(self, X_arr):
        """Helper to run prediction on best model using DataFrame to avoid feature name warnings."""
        df = pd.DataFrame(X_arr, columns=self.features)
        return self.best_model.predict(df)

    def optimize_linear(self, total_budget):
        """
        Optimize budget allocation using the Linear Regression model.
        Uses SLSQP gradient-based optimization from SciPy.
        """
        if self.lr_model is None:
            raise ValueError("Companion Linear Regression model is not trained.")

        # Objective to minimize: negative predicted sales
        def objective(x):
            # x = [TV, Radio, Newspaper]
            x_df = pd.DataFrame([x], columns=self.features)
            x_scaled = self.lr_scaler.transform(x_df)
            return -1.0 * self.lr_model.predict(x_scaled)[0]

        # Constraints: TV + Radio + Newspaper <= total_budget
        # In SciPy: fun(x) >= 0 for inequality constraints
        def budget_constraint(x):
            return total_budget - np.sum(x)

        constraints = {'type': 'ineq', 'fun': budget_constraint}

        # Bounds: spend cannot be negative, and we bound by historic maxes to avoid wild extrapolation
        bounds = [
            (0, min(300.0, total_budget)),  # TV budget bounds
            (0, min(50.0, total_budget)),   # Radio budget bounds
            (0, min(120.0, total_budget))   # Newspaper budget bounds
        ]

        # Initial guess: split budget equally
        x0 = [total_budget / 3.0] * 3

        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if not res.success:
            print(f"SLSQP Optimization warning: {res.message}")

        # Post-process allocations (rounding and clipping)
        allocations = np.clip(res.x, 0, None)
        # Re-scale to ensure we don't violate total budget due to precision
        if np.sum(allocations) > total_budget:
            allocations = (allocations / np.sum(allocations)) * total_budget
            
        predicted_sales = -res.fun

        return {
            'tv': round(float(allocations[0]), 2),
            'radio': round(float(allocations[1]), 2),
            'newspaper': round(float(allocations[2]), 2),
            'predicted_sales': round(float(predicted_sales), 2),
            'total_spend': round(float(np.sum(allocations)), 2),
            'method': 'Linear Regression (SLSQP)'
        }

    def optimize_ensemble(self, total_budget):
        """
        Optimize budget allocation using the best ensemble model (Gradient Boosting).
        Uses global Differential Evolution since tree ensembles are non-differentiable.
        """
        if not self.model_loaded:
            raise ValueError("Gradient Boosting model is not loaded.")

        # Objective to minimize: negative predicted sales
        def objective(x):
            features = np.array([x])
            if self.scaler:
                features_df = pd.DataFrame(features, columns=self.features)
                features = self.scaler.transform(features_df)
            return -1.0 * self._predict_best(features)[0]

        # Penalty-based objective to handle budget constraint
        def penalized_objective(x):
            sum_spend = np.sum(x)
            if sum_spend > total_budget:
                # Apply high quadratic penalty for budget violation
                return objective(x) + (sum_spend - total_budget) ** 2 * 1000.0
            return objective(x)

        # Bounds for each channel
        bounds = [
            (0, min(300.0, total_budget)),  # TV
            (0, min(50.0, total_budget)),   # Radio
            (0, min(120.0, total_budget))   # Newspaper
        ]

        # Run differential evolution global optimizer
        res = differential_evolution(penalized_objective, bounds, seed=42, maxiter=200, popsize=15)

        # Post-process allocations
        allocations = np.clip(res.x, 0, None)
        
        # Adjust if sum slightly exceeds budget due to optimization tolerance
        if np.sum(allocations) > total_budget:
            allocations = (allocations / np.sum(allocations)) * total_budget

        # Re-evaluate final clean prediction
        feat_pred = np.array([allocations])
        if self.scaler:
            feat_pred_df = pd.DataFrame(feat_pred, columns=self.features)
            feat_pred = self.scaler.transform(feat_pred_df)
        predicted_sales = self._predict_best(feat_pred)[0]

        return {
            'tv': round(float(allocations[0]), 2),
            'radio': round(float(allocations[1]), 2),
            'newspaper': round(float(allocations[2]), 2),
            'predicted_sales': round(float(predicted_sales), 2),
            'total_spend': round(float(np.sum(allocations)), 2),
            'method': 'Gradient Boosting (Differential Evolution)'
        }

    def predict_sales(self, tv, radio, newspaper):
        """Predict sales for a specific set of inputs using the best model."""
        if not self.model_loaded:
            raise ValueError("Model is not loaded.")
            
        features = np.array([[tv, radio, newspaper]])
        if self.scaler:
            features_df = pd.DataFrame(features, columns=self.features)
            features = self.scaler.transform(features_df)
            
        pred = self._predict_best(features)[0]
        return round(float(pred), 2)

if __name__ == "__main__":
    optimizer = BudgetOptimizer()
    
    # Test budget optimization for a total budget of $100
    budget = 100.0
    print(f"\n--- Testing Budget Optimization for ${budget} ---")
    
    try:
        lin_res = optimizer.optimize_linear(budget)
        print("Linear Model Allocation:")
        print(f"  TV: ${lin_res['tv']}, Radio: ${lin_res['radio']}, Newspaper: ${lin_res['newspaper']}")
        print(f"  Expected Sales: {lin_res['predicted_sales']} units")
    except Exception as e:
        print(f"Linear optimization failed: {e}")
        
    try:
        ens_res = optimizer.optimize_ensemble(budget)
        print("Gradient Boosting Model Allocation:")
        print(f"  TV: ${ens_res['tv']}, Radio: ${ens_res['radio']}, Newspaper: ${ens_res['newspaper']}")
        print(f"  Expected Sales: {ens_res['predicted_sales']} units")
    except Exception as e:
        print(f"Ensemble optimization failed: {e}")
