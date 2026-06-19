import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_sales_trend(df: pd.DataFrame, title: str = "Monthly Sales Trend") -> go.Figure:
    fig = px.line(
        df,
        x="Date",
        y="Sales",
        markers=True,
        title=title,
        labels={"Sales": "Sales", "Date": "Month"},
    )
    fig.update_layout(hovermode="x unified")
    return fig


def plot_actual_vs_predicted(comparison_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=comparison_df["Date"],
            y=comparison_df["Actual_Sales"],
            mode="lines+markers",
            name="Actual",
            line={"color": "#2563eb"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=comparison_df["Date"],
            y=comparison_df["Predicted_Sales"],
            mode="lines+markers",
            name="Predicted",
            line={"color": "#f97316", "dash": "dash"},
        )
    )
    fig.update_layout(
        title="Actual vs Predicted Sales (Linear Regression Fit)",
        xaxis_title="Month",
        yaxis_title="Sales",
        hovermode="x unified",
    )
    return fig


def plot_forecast_with_history(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    model_name: str,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=historical_df["Date"],
            y=historical_df["Sales"],
            mode="lines+markers",
            name="Historical Sales",
            line={"color": "#2563eb"},
        )
    )

    forecast_column = "Predicted_Sales"
    if "ARIMA_Predicted_Sales" in forecast_df.columns:
        forecast_column = "ARIMA_Predicted_Sales"

    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df[forecast_column],
            mode="lines+markers",
            name=f"{model_name} Forecast",
            line={"color": "#16a34a", "dash": "dash"},
        )
    )

    last_date = historical_df["Date"].max()
    if hasattr(last_date, "to_pydatetime"):
        last_date = last_date.to_pydatetime()

    fig.add_shape(
        type="line",
        x0=last_date,
        x1=last_date,
        y0=0,
        y1=1,
        yref="paper",
        line={"color": "#94a3b8", "dash": "dot"},
    )
    fig.add_annotation(
        x=last_date,
        y=1,
        yref="paper",
        text="Forecast starts",
        showarrow=False,
        yanchor="bottom",
        font={"color": "#64748b", "size": 11},
    )

    fig.update_layout(
        title=f"Historical Sales and {model_name} Forecast",
        xaxis_title="Month",
        yaxis_title="Sales",
        hovermode="x unified",
    )
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    valid = comparison_df.dropna(subset=["MAPE"]).copy()
    fig = px.bar(
        valid,
        x="Model",
        y="MAPE",
        title="Model Comparison (Lower MAPE Is Better)",
        text="MAPE",
        color="Model",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title="MAPE (%)")
    return fig


def plot_monthly_change(df: pd.DataFrame) -> go.Figure:
    monthly_change = df.copy()
    monthly_change["MoM_Change"] = monthly_change["Sales"].pct_change() * 100

    fig = px.bar(
        monthly_change.dropna(),
        x="Date",
        y="MoM_Change",
        title="Month-over-Month Sales Change (%)",
        labels={"MoM_Change": "Change (%)"},
        color="MoM_Change",
        color_continuous_scale=["#ef4444", "#f8fafc", "#16a34a"],
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig
