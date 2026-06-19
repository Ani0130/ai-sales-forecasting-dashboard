import pandas as pd


def calculate_kpis(df: pd.DataFrame) -> dict:
    sales = df["Sales"]
    total = sales.sum()
    average = sales.mean()
    maximum = sales.max()
    minimum = sales.min()
    latest = sales.iloc[-1]
    first = sales.iloc[0]
    growth_rate = ((latest - first) / first) * 100 if first else 0.0

    mom_change = None
    if len(sales) >= 2:
        previous = sales.iloc[-2]
        mom_change = ((latest - previous) / previous) * 100 if previous else 0.0

    return {
        "total_sales": round(total, 2),
        "average_sales": round(average, 2),
        "max_sales": round(maximum, 2),
        "min_sales": round(minimum, 2),
        "latest_sales": round(latest, 2),
        "growth_rate": round(growth_rate, 2),
        "mom_change": round(mom_change, 2) if mom_change is not None else None,
        "months_of_data": len(df),
    }


def calculate_mape(actual, predicted) -> float:
    actual = pd.Series(actual)
    predicted = pd.Series(predicted)
    mask = actual != 0
    if not mask.any():
        return 0.0
    return round(
        (abs((actual[mask] - predicted[mask]) / actual[mask]).mean()) * 100,
        2,
    )


def generate_insights(df: pd.DataFrame, kpis: dict, best_model: str, mape: float) -> list[str]:
    insights = []

    if kpis["growth_rate"] > 0:
        insights.append(
            f"Sales grew {kpis['growth_rate']:.1f}% from the first to the latest month, "
            "showing an upward business trend."
        )
    elif kpis["growth_rate"] < 0:
        insights.append(
            f"Sales declined {abs(kpis['growth_rate']):.1f}% over the recorded period. "
            "Review pricing, demand, or seasonality before relying on optimistic forecasts."
        )
    else:
        insights.append("Sales remained flat over the recorded period.")

    if kpis["mom_change"] is not None:
        direction = "increased" if kpis["mom_change"] >= 0 else "decreased"
        insights.append(
            f"The most recent month {direction} by {abs(kpis['mom_change']):.1f}% "
            "compared with the previous month."
        )

    volatility = df["Sales"].std() / df["Sales"].mean() if df["Sales"].mean() else 0
    if volatility > 0.15:
        insights.append(
            "Sales show moderate volatility. ARIMA or exponential smoothing may capture "
            "short-term swings better than a simple trend line."
        )
    else:
        insights.append(
            "Sales are relatively stable. Linear trend models should perform reasonably well."
        )

    insights.append(
        f"Based on hold-out validation, {best_model} is the strongest model "
        f"(MAPE: {mape:.1f}%)."
    )

    return insights


def format_currency(value: float) -> str:
    return f"{value:,.0f}"


def generate_business_recommendations(
    kpis: dict,
    forecast_df: pd.DataFrame,
    model_name: str,
    mape: float,
    forecast_period: int,
) -> list[str]:
    projected_total = forecast_df["Predicted_Sales"].sum()
    projected_avg = forecast_df["Predicted_Sales"].mean()
    latest = kpis["latest_sales"]
    uplift = ((projected_avg - latest) / latest) * 100 if latest else 0.0

    recommendations = [
        (
            f"Plan for approximately **{format_currency(projected_total)}** in sales "
            f"over the next **{forecast_period} months**, based on the {model_name} forecast."
        ),
    ]

    if uplift > 5:
        recommendations.append(
            f"Projected monthly sales average **{format_currency(projected_avg)}**, "
            f"about **{uplift:.1f}% above** the latest month. Consider scaling inventory, "
            "staffing, or marketing to match expected demand."
        )
    elif uplift < -5:
        recommendations.append(
            f"Forecasts suggest a **{abs(uplift):.1f}% softening** versus the latest month. "
            "Review cost structure and prioritize retention campaigns before expanding spend."
        )
    else:
        recommendations.append(
            "Forecasted demand is broadly in line with recent performance. "
            "Maintain current operations and monitor month-over-month variance."
        )

    if mape <= 10:
        recommendations.append(
            f"Model accuracy is strong (MAPE **{mape:.1f}%**). "
            "Forecasts are suitable for short-term budgeting and planning."
        )
    elif mape <= 20:
        recommendations.append(
            f"Model accuracy is moderate (MAPE **{mape:.1f}%**). "
            "Use forecasts as directional guidance and revisit after new data arrives."
        )
    else:
        recommendations.append(
            f"Model accuracy is limited (MAPE **{mape:.1f}%**). "
            "Collect more history or investigate outliers before making high-stakes decisions."
        )

    return recommendations
