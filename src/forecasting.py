from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.utils import calculate_mape

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
AVAILABLE_MODELS = {
    "linear_regression": "Linear Regression",
    "arima": "ARIMA (1,1,1)",
    "exponential_smoothing": "Exponential Smoothing",
}


def _future_dates(last_date: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=periods,
        freq="MS",
    )


def _fit_linear_regression(df: pd.DataFrame) -> LinearRegression:
    features = pd.DataFrame({"Month_Number": range(1, len(df) + 1)})
    model = LinearRegression()
    model.fit(features, df["Sales"])
    return model


def forecast_linear_regression(df: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
    model = _fit_linear_regression(df)
    future_months = pd.DataFrame(
        {"Month_Number": range(len(df) + 1, len(df) + periods + 1)}
    )
    predictions = model.predict(future_months)

    return pd.DataFrame(
        {
            "Date": _future_dates(df["Date"].max(), periods),
            "Predicted_Sales": np.maximum(predictions, 0).round(2),
            "Model": "Linear Regression",
        }
    )


def _sales_series(df: pd.DataFrame) -> pd.Series:
    series = df.set_index("Date")["Sales"].sort_index()
    return series.asfreq("MS")


def forecast_arima(df: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
    sales_series = _sales_series(df)
    model_fit = ARIMA(sales_series, order=(1, 1, 1)).fit()
    forecast = model_fit.forecast(steps=periods)

    return pd.DataFrame(
        {
            "Date": _future_dates(df["Date"].max(), periods),
            "Predicted_Sales": np.maximum(forecast.values, 0).round(2),
            "Model": "ARIMA (1,1,1)",
        }
    )


def forecast_exponential_smoothing(df: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
    sales_series = _sales_series(df)
    seasonal_periods = min(12, len(sales_series) // 2)

    if seasonal_periods >= 2:
        model = ExponentialSmoothing(
            sales_series,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods,
        )
    else:
        model = ExponentialSmoothing(sales_series, trend="add")

    fitted = model.fit(optimized=True)
    forecast = fitted.forecast(periods)

    return pd.DataFrame(
        {
            "Date": _future_dates(df["Date"].max(), periods),
            "Predicted_Sales": np.maximum(forecast.values, 0).round(2),
            "Model": "Exponential Smoothing",
        }
    )


def forecast_sales(df: pd.DataFrame, periods: int = 3, model_name: str = "linear_regression") -> pd.DataFrame:
    forecasters = {
        "linear_regression": forecast_linear_regression,
        "arima": forecast_arima,
        "exponential_smoothing": forecast_exponential_smoothing,
    }

    if model_name not in forecasters:
        raise ValueError(f"Unsupported model: {model_name}")

    return forecasters[model_name](df, periods)


def get_actual_vs_predicted(df: pd.DataFrame) -> pd.DataFrame:
    model = _fit_linear_regression(df)
    features = pd.DataFrame({"Month_Number": range(1, len(df) + 1)})
    predictions = model.predict(features)

    return pd.DataFrame(
        {
            "Date": df["Date"],
            "Actual_Sales": df["Sales"],
            "Predicted_Sales": predictions.round(2),
        }
    )


def evaluate_model(df: pd.DataFrame) -> tuple[float, float]:
    comparison = get_actual_vs_predicted(df)
    mae = mean_absolute_error(comparison["Actual_Sales"], comparison["Predicted_Sales"])
    rmse = mean_squared_error(comparison["Actual_Sales"], comparison["Predicted_Sales"]) ** 0.5
    return round(mae, 2), round(rmse, 2)


def _evaluate_on_series(actual: pd.Series, predicted: np.ndarray) -> dict:
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    mape = calculate_mape(actual, predicted)
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": mape,
    }


def evaluate_model_holdout(
    df: pd.DataFrame,
    model_name: str = "linear_regression",
    test_size: int = 3,
) -> dict:
    if len(df) <= test_size + 3:
        test_size = max(1, len(df) // 4)

    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    forecast_df = forecast_sales(train_df, periods=test_size, model_name=model_name)

    metrics = _evaluate_on_series(
        test_df["Sales"].values,
        forecast_df["Predicted_Sales"].values,
    )
    metrics["Model"] = AVAILABLE_MODELS[model_name]
    metrics["Test_Size"] = test_size
    return metrics


def compare_models(df: pd.DataFrame, test_size: int = 3) -> pd.DataFrame:
    rows = []

    for model_name in AVAILABLE_MODELS:
        try:
            metrics = evaluate_model_holdout(df, model_name=model_name, test_size=test_size)
            rows.append(metrics)
        except Exception as exc:
            rows.append(
                {
                    "Model": AVAILABLE_MODELS[model_name],
                    "MAE": None,
                    "RMSE": None,
                    "MAPE": None,
                    "Test_Size": test_size,
                    "Error": str(exc),
                }
            )

    comparison = pd.DataFrame(rows)
    valid = comparison.dropna(subset=["MAPE"])
    if not valid.empty:
        comparison = comparison.sort_values("MAPE").reset_index(drop=True)
    return comparison


def get_best_model_name(comparison_df: pd.DataFrame) -> str:
    valid = comparison_df.dropna(subset=["MAPE"])
    if valid.empty:
        return "linear_regression"

    best_label = valid.iloc[0]["Model"]
    for key, label in AVAILABLE_MODELS.items():
        if label == best_label:
            return key
    return "linear_regression"


def save_model(df: pd.DataFrame, model_name: str = "linear_regression") -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_name": model_name,
        "trained_rows": len(df),
        "last_date": df["Date"].max(),
    }

    if model_name == "linear_regression":
        artifact["model"] = _fit_linear_regression(df)
    elif model_name == "arima":
        sales_series = _sales_series(df)
        artifact["model"] = ARIMA(sales_series, order=(1, 1, 1)).fit()
    elif model_name == "exponential_smoothing":
        sales_series = _sales_series(df)
        seasonal_periods = min(12, len(sales_series) // 2)
        if seasonal_periods >= 2:
            model = ExponentialSmoothing(
                sales_series,
                trend="add",
                seasonal="add",
                seasonal_periods=seasonal_periods,
            )
        else:
            model = ExponentialSmoothing(sales_series, trend="add")
        artifact["model"] = model.fit(optimized=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    output_path = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(artifact, output_path)
    return output_path


# Backward-compatible aliases used by older code paths.
def arima_forecast_sales(df: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
    forecast_df = forecast_arima(df, periods)
    return forecast_df.rename(columns={"Predicted_Sales": "ARIMA_Predicted_Sales"})
