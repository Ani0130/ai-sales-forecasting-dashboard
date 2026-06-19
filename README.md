# AI Sales Forecasting Dashboard

An AI-powered sales analytics dashboard that analyzes historical sales trends, compares multiple forecasting models, and generates data-driven predictions to support business decision-making.

**Live Demo:** https://ai-sales-forecasting-dashboard-e6wevpy2wpq34degkbjuzo.streamlit.app/

![Dashboard Preview](assets/dashboard-preview.png)

---

## Overview

The dashboard provides an end-to-end sales forecasting pipeline, from data preprocessing and trend analysis to machine learning-based prediction and business insight generation. It allows users to upload custom sales datasets, evaluate different forecasting approaches, and visualize future sales projections through an interactive interface.

---

## Key Features

- Multi-model forecasting using Linear Regression, ARIMA, and Exponential Smoothing
- Automatic model selection based on validation performance
- Forecast evaluation using MAE, RMSE, and MAPE metrics
- Interactive sales trend, forecast, and performance visualizations
- Historical data exploration with KPI analysis
- CSV upload support for custom sales datasets
- Downloadable forecast reports
- Automated business insights and recommendations

---

## Technology Stack

- Python
- Streamlit
- Pandas & NumPy
- Scikit-learn
- Statsmodels
- Plotly
- Joblib

---

## Model Evaluation Approach

The system uses hold-out validation to evaluate forecasting accuracy. The most recent months of historical data are reserved as a test set, and each forecasting model is evaluated using MAE, RMSE, and MAPE metrics. The model with the best performance is automatically selected for future sales prediction.

---

## Project Structure

```text
ai-sales-forecasting-dashboard/
│
├── app.py                     # Streamlit application
├── requirements.txt           # Project dependencies
│
├── data/
│   ├── sales_data.csv         # Sample dataset
│   └── upload_template.csv    # CSV format reference
│
├── src/
│   ├── data_loader.py         # Data validation and preprocessing
│   ├── forecasting.py         # Forecasting models and evaluation
│   ├── visualization.py       # Interactive charts
│   └── utils.py               # KPIs and business insights
│
├── models/                    # Saved model artifacts
├── assets/                    # Dashboard images
└── .streamlit/                # Streamlit configuration
```

---

## Future Enhancements

- Forecast confidence intervals
- Product and region-level forecasting
- Integration with databases and external APIs
- Automated periodic model retraining

---

## Author

**Aniket Jha**

GitHub: https://github.com/Ani0130  
LinkedIn: https://www.linkedin.com/in/aniket-jha-1051b4252/

---

## License

MIT