# 📊 Sales Prediction using Python
This sub-project predicts future sales based on advertising budgets spent on **TV**, **Radio**, and **Newspaper** channels.

---

## 📁 Directory Structure
```
sales_prediction/
├── data/
│   └── Advertising.csv       # Dataset containing advertising budgets and sales
├── models/
│   └── sales_model.pkl       # Serialized best-performing ML model (Gradient Boosting)
├── plots/
│   ├── actual_vs_predicted.png
│   ├── advertising_impact.png
│   ├── correlation_matrix.png
│   ├── model_comparison.png
│   ├── residuals_distribution.png
│   └── spend_vs_sales.png
├── public/                   # Web application frontend
│   ├── index.html            /* Interactive dashboard layout */
│   ├── style.css             /* Translucent glassmorphic dark styling */
│   └── app.js                /* AJAX endpoints integration & Chart.js plots */
├── src/
│   ├── app.py                # FastAPI web server and API endpoints
│   ├── create_notebook.py    # Script to generate the Jupyter Notebook
│   ├── optimize.py           # Budget allocation optimization engine (SciPy)
│   └── train.py              # Modular training pipeline
├── README.md                 # Project instructions
├── Sales_Prediction.ipynb    # Jupyter Notebook containing full EDA & modeling
└── requirements.txt          # Python dependencies
```

---

## 🚀 Setup & Execution

### 1. Install Dependencies
Make sure you have Python 3.8+ installed. You can install all required dependencies using:
```bash
pip install -r requirements.txt
```

### 2. Run the Training Pipeline
To run exploratory data analysis, train models, select the best model, save it, and update plots, execute:
```bash
python src/train.py
```

### 3. Launch the Interactive Web Dashboard
To start the FastAPI server and explore prediction and budget optimization interactively:
```bash
python -m uvicorn src.app:app --reload
```
Once the server starts, open your browser and navigate to **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

---

## 📈 Model Performance Summary

| Model | Test $R^2$ Score | Test MAE | Test RMSE | 5-Fold CV $R^2$ |
| :--- | :--- | :--- | :--- | :--- |
| **Gradient Boosting** | **0.9831** | **0.6187** | **0.7298** | **0.9775** |
| **Random Forest** | 0.9813 | 0.6201 | 0.7686 | 0.9755 |
| **Linear Regression** | 0.8994 | 1.4608 | 1.7816 | 0.8871 |
| **Ridge Regression** | 0.8988 | 1.4643 | 1.7872 | 0.8871 |
| **Lasso Regression** | 0.8983 | 1.4613 | 1.7913 | 0.8886 |

*   **Best Model**: **Gradient Boosting Regressor** ($R^2 = 98.31\%$, $RMSE = 0.730$).
*   **Key Insight**: TV spend is the most significant predictor (61.8% relative importance), followed by Radio spend (34.6% relative importance). Newspaper advertising has negligible impact on sales.
