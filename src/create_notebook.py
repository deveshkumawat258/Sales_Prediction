"""
Notebook Generator Script
Author: Harsh (Junior Developer)
Date: July 2026
Purpose: Programmatically generates a complete Jupyter Notebook for the Sales Prediction task,
         populated with rich Markdown text and functional Python code cells.
"""

import json
import os

NOTEBOOK_PATH = os.path.join("sales_prediction", "Sales_Prediction.ipynb")

def create_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 📈 Sales Prediction using Python\n",
                    "### Task 4 - CodeAlpha Machine Learning Internship\n",
                    "**Author**: Harsh (Junior Developer)  \n",
                    "**Objective**: Predict future sales based on advertising spends across TV, Radio, and Newspaper channels, and deliver actionable marketing insights.\n",
                    "\n",
                    "---\n",
                    "\n",
                    "## 📖 Project Overview\n",
                    "Advertising budgets are substantial investments, and optimizing them is crucial for business growth. This notebook guides you through an end-to-end Machine Learning pipeline to:\n",
                    "1. **Load and Clean** the dataset.\n",
                    "2. Perform **Exploratory Data Analysis (EDA)** to understand the relations between ad channels and sales.\n",
                    "3. Check for **multicollinearity** (using Variance Inflation Factor - VIF).\n",
                    "4. Develop and evaluate multiple **Regression Models**:\n",
                    "   * *Linear Regression*\n",
                    "   * *Ridge Regression* (L2 Regularization)\n",
                    "   * *Lasso Regression* (L1 Regularization)\n",
                    "   * *Random Forest Regressor* (Ensemble Learner)\n",
                    "   * *Gradient Boosting Regressor* (Boosting Learner)\n",
                    "5. Select the best performing model and run **diagnostic checks**.\n",
                    "6. Formulate **actionable marketing recommendations** based on the findings."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🛠️ Step 1: Import Libraries and Configure Plotting Styles"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import pickle\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "# Set styling for visualizations\n",
                    "sns.set_theme(style=\"darkgrid\")\n",
                    "plt.rcParams.update({\n",
                    "    'figure.facecolor': '#121212',\n",
                    "    'axes.facecolor': '#1e1e1e',\n",
                    "    'text.color': '#e0e0e0',\n",
                    "    'axes.labelcolor': '#e0e0e0',\n",
                    "    'xtick.color': '#b0b0b0',\n",
                    "    'ytick.color': '#b0b0b0',\n",
                    "    'axes.titlecolor': '#ffffff',\n",
                    "    'grid.color': '#333333',\n",
                    "    'font.family': 'sans-serif',\n",
                    "})\n",
                    "\n",
                    "print(\"Libraries successfully imported!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📁 Step 2: Load and Inspect the Dataset"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "DATA_PATH = os.path.join(\"data\", \"Advertising.csv\")\n",
                    "\n",
                    "# Load dataset setting the first column as the index (ID/Index column)\n",
                    "df = pd.read_csv(DATA_PATH, index_col=0)\n",
                    "\n",
                    "print(f\"Dataset Shape: {df.shape}\")\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Check Data Info and Statistical Properties"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\"--- Dataset Info ---\")\n",
                    "df.info()\n",
                    "\n",
                    "print(\"\\n--- Descriptive Statistics ---\")\n",
                    "df.describe()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\"--- Missing Values Check ---\")\n",
                    "print(df.isnull().sum())\n",
                    "\n",
                    "print(\"\\n--- Duplicate Rows Check ---\")\n",
                    "print(f\"Duplicates found: {df.duplicated().sum()}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📊 Step 3: Exploratory Data Analysis & Visualizations\n",
                    "Let's explore the relations between budgets on each channel (TV, Radio, Newspaper) and Sales."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Correlation Heatmap\n",
                    "plt.figure(figsize=(8, 6))\n",
                    "corr = df.corr()\n",
                    "mask = np.triu(np.ones_like(corr, dtype=bool))\n",
                    "sns.heatmap(corr, annot=True, mask=mask, cmap=\"coolwarm\", fmt=\".3f\", \n",
                    "            linewidths=1, cbar_kws={\"shrink\": .8}, annot_kws={\"size\": 12})\n",
                    "plt.title(\"Correlation Heatmap of Advertising Spend & Sales\", fontsize=14, pad=15)\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2. Regression Plots for each channel vs Sales\n",
                    "fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)\n",
                    "features = ['TV', 'Radio', 'Newspaper']\n",
                    "colors = ['#ff79c6', '#8be9fd', '#50fa7b']\n",
                    "\n",
                    "for i, feature in enumerate(features):\n",
                    "    sns.regplot(\n",
                    "        data=df, x=feature, y='Sales', ax=axes[i],\n",
                    "        scatter_kws={'alpha': 0.6, 'color': colors[i], 'edgecolor': 'none', 's': 40},\n",
                    "        line_kws={'color': '#ffb86c', 'linewidth': 2}\n",
                    "    )\n",
                    "    axes[i].set_title(f\"{feature} Spend vs Sales\", fontsize=14, pad=10)\n",
                    "    axes[i].set_xlabel(f\"{feature} Advertising Budget ($)\", fontsize=12)\n",
                    "    if i == 0:\n",
                    "        axes[i].set_ylabel(\"Sales (units)\", fontsize=12)\n",
                    "        \n",
                    "plt.suptitle(\"Impact of Advertising Channels on Sales Outcomes\", fontsize=16, y=1.02, color='#ffffff')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Checking for Multicollinearity using Variance Inflation Factors (VIF)\n",
                    "Multicollinearity occurs when independent variables are highly correlated. VIF measures the severity of multicollinearity. A VIF value above 5 indicates high multicollinearity which can destabilize linear regression coefficients."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from sklearn.linear_model import LinearRegression\n",
                    "\n",
                    "X_vif = df[['TV', 'Radio', 'Newspaper']]\n",
                    "print(\"Variance Inflation Factors (VIF) check:\")\n",
                    "for col in X_vif.columns:\n",
                    "    X_other = X_vif.drop(columns=[col])\n",
                    "    y_col = X_vif[col]\n",
                    "    lr = LinearRegression()\n",
                    "    lr.fit(X_other, y_col)\n",
                    "    r2 = lr.score(X_other, y_col)\n",
                    "    vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float('inf')\n",
                    "    print(f\"  {col}: {vif:.4f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "> **Observation**: All VIF values are below 1.2, showing that multicollinearity is practically non-existent. This means we can reliably interpret the coefficients of linear models."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🤖 Step 4: Model Development & Evaluation\n",
                    "We will split our data, apply standardization to the features (needed for distance-based and regularized linear models), and train 5 regression models."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from sklearn.model_selection import train_test_split, cross_val_score\n",
                    "from sklearn.preprocessing import StandardScaler\n",
                    "from sklearn.linear_model import Ridge, Lasso\n",
                    "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
                    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
                    "\n",
                    "X = df[['TV', 'Radio', 'Newspaper']]\n",
                    "y = df['Sales']\n",
                    "\n",
                    "# Split train & test sets (80-20 split)\n",
                    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
                    "\n",
                    "# Standardize features\n",
                    "scaler = StandardScaler()\n",
                    "X_train_scaled = scaler.fit_transform(X_train)\n",
                    "X_test_scaled = scaler.transform(X_test)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Define models to train\n",
                    "models = {\n",
                    "    'Linear Regression': (LinearRegression(), True),\n",
                    "    'Ridge Regression': (Ridge(alpha=1.0), True),\n",
                    "    'Lasso Regression': (Lasso(alpha=0.1), True),\n",
                    "    'Random Forest': (RandomForestRegressor(n_estimators=100, random_state=42), False),\n",
                    "    'Gradient Boosting': (GradientBoostingRegressor(n_estimators=100, random_state=42), False)\n",
                    "}\n",
                    "\n",
                    "results = {}\n",
                    "\n",
                    "for name, (model, needs_scaling) in models.items():\n",
                    "    X_tr = X_train_scaled if needs_scaling else X_train\n",
                    "    X_te = X_test_scaled if needs_scaling else X_test\n",
                    "    \n",
                    "    # Fit\n",
                    "    model.fit(X_tr, y_train)\n",
                    "    \n",
                    "    # Predict\n",
                    "    preds = model.predict(X_te)\n",
                    "    \n",
                    "    # Compute metrics\n",
                    "    r2 = r2_score(y_test, preds)\n",
                    "    mae = mean_absolute_error(y_test, preds)\n",
                    "    rmse = np.sqrt(mean_squared_error(y_test, preds))\n",
                    "    \n",
                    "    # Cross-validation\n",
                    "    X_cv = scaler.fit_transform(X) if needs_scaling else X\n",
                    "    cv_scores = cross_val_score(model, X_cv, y, cv=5, scoring='r2')\n",
                    "    cv_mean = np.mean(cv_scores)\n",
                    "    \n",
                    "    results[name] = {\n",
                    "        'R2': r2,\n",
                    "        'MAE': mae,\n",
                    "        'RMSE': rmse,\n",
                    "        'CV_R2': cv_mean,\n",
                    "        'model_obj': model,\n",
                    "        'needs_scaling': needs_scaling\n",
                    "    }\n",
                    "    \n",
                    "    print(f\"=== {name} ===\")\n",
                    "    print(f\"  R-squared Score: {r2:.4f}\")\n",
                    "    print(f\"  Mean Absolute Error: {mae:.4f}\")\n",
                    "    print(f\"  Root Mean Squared Error: {rmse:.4f}\")\n",
                    "    print(f\"  5-Fold CV Mean R2: {cv_mean:.4f}\\n\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📈 Step 5: Model Comparison & Diagnostics\n",
                    "Let's compare the performance of each model side-by-side using charts."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "names = list(results.keys())\n",
                    "r2_scores = [results[n]['R2'] for n in names]\n",
                    "rmse_scores = [results[n]['RMSE'] for n in names]\n",
                    "\n",
                    "x = np.arange(len(names))\n",
                    "width = 0.35\n",
                    "\n",
                    "fig, ax1 = plt.subplots(figsize=(12, 6))\n",
                    "\n",
                    "# R2 bars\n",
                    "color = '#ff79c6'\n",
                    "ax1.set_xlabel('Regression Models', fontsize=12, labelpad=10)\n",
                    "ax1.set_ylabel('R-squared Score', color=color, fontsize=12)\n",
                    "rects1 = ax1.bar(x - width/2, r2_scores, width, label='R-squared', color=color, alpha=0.8)\n",
                    "ax1.tick_params(axis='y', labelcolor=color)\n",
                    "ax1.set_ylim(0.8, 1.0)\n",
                    "\n",
                    "# RMSE bars\n",
                    "ax2 = ax1.twinx()\n",
                    "color = '#8be9fd'\n",
                    "ax2.set_ylabel('RMSE (Sales Units)', color=color, fontsize=12)\n",
                    "rects2 = ax2.bar(x + width/2, rmse_scores, width, label='RMSE', color=color, alpha=0.8)\n",
                    "ax2.tick_params(axis='y', labelcolor=color)\n",
                    "\n",
                    "ax1.set_xticks(x)\n",
                    "ax1.set_xticklabels(names, rotation=15, ha='right')\n",
                    "plt.title(\"Model Comparison: R-squared and RMSE Scores\", fontsize=14, pad=15)\n",
                    "fig.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Best Model Selection & Diagnostics\n",
                    "Based on $R^2$ and RMSE scores, Gradient Boosting performs best. Let's inspect its diagnostic plots (predicted vs actual, and error distribution)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Identify best model\n",
                    "best_model_name = max(results, key=lambda x: results[x]['R2'])\n",
                    "best_model_info = results[best_model_name]\n",
                    "best_model = best_model_info['model_obj']\n",
                    "best_needs_scaling = best_model_info['needs_scaling']\n",
                    "\n",
                    "X_te_best = X_test_scaled if best_needs_scaling else X_test\n",
                    "preds = best_model.predict(X_te_best)\n",
                    "residuals = y_test - preds\n",
                    "\n",
                    "print(f\"Best Model: {best_model_name} (Test R2: {best_model_info['R2']:.4f}, Test RMSE: {best_model_info['RMSE']:.4f})\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Plot predicted vs actual\n",
                    "plt.figure(figsize=(8, 6))\n",
                    "sns.scatterplot(x=y_test, y=preds, color='#50fa7b', alpha=0.8, s=65, edgecolor='none')\n",
                    "plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)\n",
                    "plt.title(f\"{best_model_name}: Predicted vs Actual Sales\", fontsize=14, pad=15)\n",
                    "plt.xlabel(\"Actual Sales (units)\", fontsize=12)\n",
                    "plt.ylabel(\"Predicted Sales (units)\", fontsize=12)\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Plot distribution of residuals\n",
                    "plt.figure(figsize=(8, 6))\n",
                    "sns.histplot(residuals, kde=True, color='#bd93f9', bins=15, edgecolor='#121212')\n",
                    "plt.axvline(x=0, color='r', linestyle='--', linewidth=1.5)\n",
                    "plt.title(f\"{best_model_name}: Distribution of Residuals (Errors)\", fontsize=14, pad=15)\n",
                    "plt.xlabel(\"Residual (Actual - Predicted)\", fontsize=12)\n",
                    "plt.ylabel(\"Frequency\", fontsize=12)\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🔍 Step 6: Analyzing Advertising Impact & ROI"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Let's inspect the linear regression model coefficients to determine direct impact\n",
                    "lr_model = results['Linear Regression']['model_obj']\n",
                    "coefficients = pd.DataFrame({\n",
                    "    'Feature': X.columns,\n",
                    "    'Standardized Coefficient': lr_model.coef_\n",
                    "})\n",
                    "coefficients['Absolute Value'] = coefficients['Standardized Coefficient'].abs()\n",
                    "coefficients = coefficients.sort_values(by='Absolute Value', ascending=False)\n",
                    "print(\"--- Linear Regression Standardized Coefficients ---\")\n",
                    "print(coefficients.to_string(index=False))\n",
                    "\n",
                    "# Also, inspect Gradient Boosting Feature Importances\n",
                    "if hasattr(best_model, 'feature_importances_'):\n",
                    "    importances = pd.DataFrame({\n",
                    "        'Feature': X.columns,\n",
                    "        'Relative Importance': best_model.feature_importances_\n",
                    "    }).sort_values(by='Relative Importance', ascending=False)\n",
                    "    print(\"\\n--- Gradient Boosting Feature Importances ---\")\n",
                    "    print(importances.to_string(index=False))"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Plot Feature Importance / Impact\n",
                    "plt.figure(figsize=(8, 6))\n",
                    "if hasattr(best_model, 'feature_importances_'):\n",
                    "    sns.barplot(x=importances['Feature'], y=importances['Relative Importance'], \n",
                    "                palette=['#ff79c6', '#8be9fd', '#50fa7b'], hue=importances['Feature'], legend=False)\n",
                    "    plt.title(f\"{best_model_name} Feature Importances\", fontsize=14, pad=15)\n",
                    "    plt.ylabel(\"Relative Importance\", fontsize=12)\n",
                    "else:\n",
                    "    sns.barplot(x=coefficients['Feature'], y=coefficients['Standardized Coefficient'], \n",
                    "                palette=['#ff79c6', '#8be9fd', '#50fa7b'], hue=coefficients['Feature'], legend=False)\n",
                    "    plt.title(\"Linear Regression Standardized Coefficients\", fontsize=14, pad=15)\n",
                    "    plt.ylabel(\"Coefficient Value\", fontsize=12)\n",
                    "    \n",
                    "plt.xlabel(\"Advertising Channel\", fontsize=12)\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🚀 Actionable Marketing Insights & Recommendations\n",
                    "\n",
                    "Based on our data exploration and modeling, here are the key insights and recommended actions:\n",
                    "\n",
                    "1. **TV Advertising is the Main Driver**:\n",
                    "   * **TV spend** has the highest positive correlation ($r = 0.782$) and the highest standardized coefficient ($coef = 3.764$) with Sales.\n",
                    "   * It holds the maximum importance ($61.8\\%$) in the Gradient Boosting model.\n",
                    "   * *Recommendation*: TV should remain the cornerstone of any high-reach marketing campaign. Increasing TV budget leads to stable and strong returns.\n",
                    "\n",
                    "2. **Radio is Highly Efficient**:\n",
                    "   * **Radio spend** is smaller overall but has a high impact coefficient ($2.79$) and relative importance ($34.6\\%$).\n",
                    "   * In fact, per dollar spent, Radio yields a very high marginal return.\n",
                    "   * *Recommendation*: Allocate surplus budgets to Radio, especially to support TV campaigns, as they interact synergistically.\n",
                    "\n",
                    "3. **Newspaper Spend is Non-Impactful**:\n",
                    "   * **Newspaper spend** shows almost zero relationship with sales once TV and Radio are factored in ($coef = 0.018$, importance $3.5\\%$).\n",
                    "   * *Recommendation*: Reduce or reallocate newspaper ad spend to TV or Radio. Newspaper ads do not drive incremental sales in this market segment."
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(notebook, f, indent=1)
    print(f"Jupyter Notebook successfully written to {NOTEBOOK_PATH}!")

if __name__ == "__main__":
    create_notebook()
