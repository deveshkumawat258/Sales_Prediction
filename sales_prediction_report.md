# 📋 Executive Report: Sales Prediction & Advertising Spend Optimization
**Prepared for**: Senior Marketing Leadership  
**Author**: Harsh (Junior Developer, Antigravity AI)  
**Date**: July 2026  
**Objective**: Build a robust machine learning pipeline to predict future sales, analyze the marginal impact of TV, Radio, and Newspaper advertising channels, and deliver data-driven budget optimization strategies.

---

## 🎯 Executive Summary
Using historical campaign data of **200 marketing cycles**, we developed a highly accurate predictive framework to forecast sales outcomes. Our evaluation of linear and tree-based machine learning architectures revealed that the **Gradient Boosting Regressor** predicts sales with **98.31% accuracy** ($R^2 = 0.9831$).

The core findings indicate a strong disparity in return on investment (ROI) across advertising mediums:
1.  **TV Advertising** is the primary driver of total sales volume, contributing 61.8% of predictive weight.
2.  **Radio Advertising** is highly efficient, generating high marginal sales per dollar spent.
3.  **Newspaper Advertising** yields negligible incremental sales, signifying a key area for cost-savings and budget reallocation.

---

## 📊 Exploratory Data Analysis & Visualizations
We analyzed the correlation structures and individual regression lines to understand how spending on each channel affects sales.

### 1. Correlation Analysis
The correlation matrix heatmap shows the pairwise linear relationships between advertising channels and sales:
*   **TV and Sales**: Extremely strong positive linear relationship ($r = 0.782$).
*   **Radio and Sales**: Moderate-to-strong positive relationship ($r = 0.576$).
*   **Newspaper and Sales**: Weak positive relationship ($r = 0.228$).
*   **Inter-channel Correlation**: Correlations between the advertising channels themselves are extremely low (all $r < 0.35$), suggesting that spending on one channel does not overlap with another.

> **Visual Evidence**: See the correlation details in [correlation_matrix.png](file:///c:/Users/Harsh/Desktop/CodeAlpha/sales_prediction/plots/correlation_matrix.png).

### 2. Advertising Channel Impact Curves
To visualize the marginal impact of budget increases, we plotted scatter plots with fitted linear regression lines for each channel against sales:
*   **TV**: Shows a steady, positive, linear progression from 0 to 300 units of budget.
*   **Radio**: Displays a steep positive slope, suggesting that even small budget increments in Radio drive significant sales increases.
*   **Newspaper**: Displays a flat regression line with highly scattered data points, indicating that newspaper spending is a poor predictor of sales.

> **Visual Evidence**: View individual regression fits in [spend_vs_sales.png](file:///c:/Users/Harsh/Desktop/CodeAlpha/sales_prediction/plots/spend_vs_sales.png).

### 3. Multicollinearity Assessment (VIF)
To ensure the regression coefficients are stable and interpretable, we calculated the **Variance Inflation Factor (VIF)**:
*   **TV VIF**: $1.0046$
*   **Radio VIF**: $1.1450$
*   **Newspaper VIF**: $1.1452$

All VIF values are well below the threshold of $5.0$ (in fact, they are close to the ideal $1.0$), proving there is no multicollinearity in this dataset.

---

## 🤖 Model Development & Comparison
We trained five different regression algorithms using an 80-20 train-test split. The models were evaluated using $R^2$ (coefficient of determination), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE). We also ran 5-fold cross-validation to assess generalizability.

### Model Performance Metrics Table

| Model | Test $R^2$ Score | Test MAE | Test RMSE | 5-Fold CV $R^2$ Mean |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **0.9831** | **0.6187** | **0.7298** | **0.9775** |
| **Random Forest** | 0.9813 | 0.6201 | 0.7686 | 0.9755 |
| **Linear Regression** | 0.8994 | 1.4608 | 1.7816 | 0.8871 |
| **Ridge Regression** | 0.8988 | 1.4643 | 1.7872 | 0.8871 |
| **Lasso Regression** | 0.8983 | 1.4613 | 1.7913 | 0.8886 |

> **Visual Evidence**: See the comparative performance chart in [model_comparison.png](file:///c:/Users/Harsh/Desktop/CodeAlpha/sales_prediction/plots/model_comparison.png).

### Key Technical Insights:
*   **Linear vs. Non-linear Models**: Tree-based ensemble models (Gradient Boosting and Random Forest) outperform linear models by a large margin, boosting the $R^2$ score from $\approx 90\%$ to over **$98\%$** and cutting error rates (RMSE) by **59%**.
*   **Interaction Effects**: This performance jump indicates that there are non-linear interaction effects between marketing channels. For example, spending on TV and Radio simultaneously has a synergistic effect on sales that a standard linear model cannot capture without explicit interaction terms.

---

## 🔍 Best Model Evaluation & Diagnostics
The **Gradient Boosting Regressor** was selected as our production-grade model. We conducted rigorous diagnostic evaluations:

### 1. Actual vs. Predicted Sales
A scatter plot of actual sales against the model's predictions shows the points tightly clustered along the 45-degree diagonal line $y = x$. This indicates that the model predicts sales with minimal variance across all sales brackets.
> **Visual Evidence**: View the actual vs. predicted plot in [actual_vs_predicted.png](file:///c:/Users/Harsh/Desktop/CodeAlpha/sales_prediction/plots/actual_vs_predicted.png).

### 2. Residual Analysis
A histogram of the prediction errors (residuals) shows a symmetric, bell-shaped distribution centered around $0.0$. This normal distribution of residuals indicates that the model's errors are random noise and there are no systematic patterns or biases left uncaptured.
> **Visual Evidence**: View the residuals distribution plot in [residuals_distribution.png](file:///c:/Users/Harsh/Desktop/CodeAlpha/sales_prediction/plots/residuals_distribution.png).

### 3. Feature Importance Analysis
We extracted the relative feature importances from the Gradient Boosting model:
*   **TV**: **$61.83\%$** relative importance.
*   **Radio**: **$34.64\%$** relative importance.
*   **Newspaper**: **$3.53\%$** relative importance.

> **Visual Evidence**: View the feature importance chart in [advertising_impact.png](file:///c:/Users/Harsh/Desktop/CodeAlpha/sales_prediction/plots/advertising_impact.png).

---

## 🚀 Actionable Marketing Strategies

Based on the empirical findings, we recommend the following strategic reallocations:

### 1. Prioritize TV for Baseline Volume
*   **Finding**: TV budget accounts for the majority of sales volume. Every $1 increase in standardized TV spend is associated with a $3.76 unit increase in sales in a linear setup, and accounts for 61.8% of the predictive power.
*   **Strategy**: Maintain TV advertising as the foundation of the brand's reach campaigns. Ensure consistent baseline spending on TV to sustain high volume sales.

### 2. Capitalize on Radio for High Marginal Returns
*   **Finding**: Radio has an extremely high relative impact ($34.6\%$ importance) despite having a smaller budget footprint in the raw data. Per dollar, its slope is very steep.
*   **Strategy**: Increase the budget allocation for Radio. Radio works best as an accelerator, reinforcing TV campaign messaging at a lower cost per impression.

### 3. Defund Newspaper Campaigns
*   **Finding**: Newspaper spend has a correlation of only $0.228$ with Sales, a standardized linear coefficient of just $0.018$ (almost zero), and a relative importance of only $3.5\%$ in Gradient Boosting. It provides no statistically significant incremental returns.
*   **Strategy**: Defund print Newspaper advertising entirely. Reallocate 100% of the Newspaper budget to TV and Radio to maximize sales revenue without increasing overall marketing expenditure.

---

## 🏁 Conclusion
By migrating to a non-linear **Gradient Boosting** framework, the organization can forecast sales with **98.3% accuracy**. Implementing the budget reallocation strategy—specifically defunding Newspaper and boosting Radio/TV allocations—is projected to significantly increase overall sales yields while keeping the total marketing budget flat.

The trained model has been saved as a serialized pickle file in [sales_model.pkl](file:///c:/Users/Harsh/Desktop/CodeAlpha/sales_prediction/models/sales_model.pkl) for future forecasting and scenario analysis.
