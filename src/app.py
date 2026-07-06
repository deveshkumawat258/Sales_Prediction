"""
FastAPI Server for Sales Prediction App
Author: Harsh (Junior Developer)
Date: July 2026
Purpose: Provides API endpoints for predicting sales and optimizing budgets,
         and serves the static web dashboard.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sales_prediction.src.optimize import BudgetOptimizer

app = FastAPI(
    title="Sales Prediction API",
    description="API for predicting sales based on advertising budgets and optimizing spend.",
    version="1.0.0"
)

# Enable CORS for local testing if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Optimizer
optimizer = BudgetOptimizer()

# Define request schemas
class PredictRequest(BaseModel):
    tv: float = Field(..., description="TV advertising budget in thousands of dollars", ge=0)
    radio: float = Field(..., description="Radio advertising budget in thousands of dollars", ge=0)
    newspaper: float = Field(..., description="Newspaper advertising budget in thousands of dollars", ge=0)

class OptimizeRequest(BaseModel):
    total_budget: float = Field(..., description="Total available advertising budget", gt=0)

@app.post("/api/predict")
def predict_sales_endpoint(data: PredictRequest):
    """Predict expected sales units for given TV, Radio, and Newspaper budgets."""
    try:
        sales = optimizer.predict_sales(data.tv, data.radio, data.newspaper)
        return {
            "success": True,
            "tv": data.tv,
            "radio": data.radio,
            "newspaper": data.newspaper,
            "predicted_sales_units": sales
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/optimize")
def optimize_budget_endpoint(data: OptimizeRequest):
    """
    Find optimal budget allocations to maximize sales.
    Returns recommendations for both Linear Regression and Gradient Boosting models.
    """
    try:
        # Get allocations for both models
        linear_alloc = optimizer.optimize_linear(data.total_budget)
        ensemble_alloc = optimizer.optimize_ensemble(data.total_budget)
        
        return {
            "success": True,
            "total_budget": data.total_budget,
            "linear_model_allocation": linear_alloc,
            "ensemble_model_allocation": ensemble_alloc
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

@app.get("/api/insights")
def get_insights_endpoint():
    """Retrieve marketing channel insights, coefficients, and relative importances."""
    # Standardized linear regression coefficients
    linear_coefs = {
        "TV": 3.76,
        "Radio": 2.79,
        "Newspaper": 0.02
    }
    
    # Gradient boosting relative importances
    gradient_boosting_importances = {
        "TV": 0.618,
        "Radio": 0.346,
        "Newspaper": 0.035
    }
    
    # Historic correlation with Sales
    correlations = {
        "TV": 0.782,
        "Radio": 0.576,
        "Newspaper": 0.228
    }

    return {
        "success": True,
        "linear_coefficients": linear_coefs,
        "ensemble_importances": gradient_boosting_importances,
        "sales_correlations": correlations
    }

# Mount static files folder to serve the frontend web page
# The public/ folder must exist. It will contain index.html, style.css, app.js
PUBLIC_DIR = os.path.join("sales_prediction", "public")
if not os.path.exists(PUBLIC_DIR):
    os.makedirs(PUBLIC_DIR, exist_ok=True)

app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sales_prediction.src.app:app", host="127.0.0.1", port=8000, reload=True)
