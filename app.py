from pathlib import Path

import streamlit as st

from src.data_loader import load_data, load_uploaded_file
from src.forecasting import (
    AVAILABLE_MODELS,
    compare_models,
    forecast_sales,
    get_actual_vs_predicted,
    get_best_model_name,
    save_model,
)
from src.utils import calculate_kpis, generate_business_recommendations, generate_insights
from src.visualization import (
    plot_actual_vs_predicted,
    plot_forecast_with_history,
    plot_model_comparison,
    plot_monthly_change,
    plot_sales_trend,
)

TEMPLATE_PATH = Path(__file__).resolve().parent / "data" / "upload_template.csv"

st.set_page_config(
    page_title="AI Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
)

hero_left, hero_right = st.columns([3, 1])
with hero_left:
    st.title("AI Sales Forecasting Dashboard")
    st.markdown(
        "End-to-end ML analytics app for **sales trend analysis**, **multi-model forecasting**, "
        "and **data-driven business recommendations**."
    )
with hero_right:
    st.markdown("**Built with**")
    st.markdown("`Python` `Streamlit` `Scikit-learn` `Statsmodels` `Plotly`")


@st.cache_data
def get_sample_data():
    return load_data()


def load_dashboard_data():
    uploaded_file = st.session_state.get("uploaded_file")
    if uploaded_file is not None:
        return load_uploaded_file(uploaded_file)
    return get_sample_data()


with st.sidebar:
    st.header("Controls")

    uploaded_file = st.file_uploader(
        "Upload sales CSV",
        type=["csv"],
        help="CSV must include Date and Sales columns (YYYY-MM-DD recommended).",
    )
    st.session_state["uploaded_file"] = uploaded_file

    if uploaded_file is None:
        st.info("Using bundled sample dataset. Upload a CSV to analyze your own sales.")
    else:
        st.success(f"Loaded file: {uploaded_file.name}")

    st.download_button(
        label="Download CSV template",
        data=TEMPLATE_PATH.read_bytes(),
        file_name="upload_template.csv",
        mime="text/csv",
    )

    forecast_period = st.slider("Forecast horizon (months)", 1, 12, 3)
    test_size = st.slider("Hold-out test months", 1, 6, 3)

    st.divider()
    model_choice = st.selectbox(
        "Primary forecast model",
        options=list(AVAILABLE_MODELS.keys()),
        format_func=lambda key: AVAILABLE_MODELS[key],
    )
    use_best_model = st.checkbox("Use best model from validation", value=True)

    with st.expander("About this project"):
        st.markdown(
            """
            Portfolio project showcasing:

            - Time-series forecasting with 3 ML/statistical models
            - Hold-out validation using MAPE, MAE, and RMSE
            - Interactive analytics UI with exports and model persistence

            Portfolio project by **Aniket Jha** — time-series forecasting with
            validated ML models and interactive analytics.
            """
        )

try:
    df = load_dashboard_data()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

comparison_df = compare_models(df, test_size=test_size)
selected_model = get_best_model_name(comparison_df) if use_best_model else model_choice
forecast_df = forecast_sales(df, periods=forecast_period, model_name=selected_model)
kpis = calculate_kpis(df)
best_mape = (
    comparison_df.dropna(subset=["MAPE"]).iloc[0]["MAPE"]
    if not comparison_df.dropna(subset=["MAPE"]).empty
    else 0.0
)
insights = generate_insights(df, kpis, AVAILABLE_MODELS[selected_model], best_mape)
recommendations = generate_business_recommendations(
    kpis,
    forecast_df,
    AVAILABLE_MODELS[selected_model],
    best_mape,
    forecast_period,
)

overview_tab, data_tab, forecast_tab, performance_tab = st.tabs(
    ["Overview", "Data Explorer", "Forecasting", "Model Performance"]
)

with overview_tab:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Sales", f"{kpis['total_sales']:,.0f}")
    col2.metric("Average Monthly Sales", f"{kpis['average_sales']:,.0f}")
    col3.metric("Latest Month", f"{kpis['latest_sales']:,.0f}")
    col4.metric("Overall Growth", f"{kpis['growth_rate']:.1f}%")
    col5.metric("Best Model MAPE", f"{best_mape:.1f}%")

    rec_col, insight_col = st.columns(2)
    with rec_col:
        st.subheader("Business Recommendations")
        for recommendation in recommendations:
            st.markdown(f"- {recommendation}")
    with insight_col:
        st.subheader("Executive Insights")
        for insight in insights:
            st.write(f"- {insight}")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(plot_sales_trend(df), width="stretch")
    with chart_col2:
        st.plotly_chart(plot_monthly_change(df), width="stretch")

with data_tab:
    st.subheader("Sales Dataset")
    st.dataframe(df, width="stretch", hide_index=True)

    st.download_button(
        label="Download historical data",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="historical_sales.csv",
        mime="text/csv",
    )

    st.plotly_chart(plot_sales_trend(df, title="Interactive Sales History"), width="stretch")

with forecast_tab:
    st.subheader(f"Forecast using {AVAILABLE_MODELS[selected_model]}")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    projected_total = forecast_df["Predicted_Sales"].sum()
    projected_avg = forecast_df["Predicted_Sales"].mean()
    projected_peak = forecast_df["Predicted_Sales"].max()

    metric_col1.metric("Projected total", f"{projected_total:,.0f}")
    metric_col2.metric("Projected average", f"{projected_avg:,.0f}")
    metric_col3.metric("Projected peak month", f"{projected_peak:,.0f}")

    st.plotly_chart(
        plot_forecast_with_history(df, forecast_df, AVAILABLE_MODELS[selected_model]),
        width="stretch",
    )

    st.dataframe(forecast_df, width="stretch", hide_index=True)

    export_col1, export_col2 = st.columns(2)
    with export_col1:
        st.download_button(
            label="Download forecast CSV",
            data=forecast_df.to_csv(index=False).encode("utf-8"),
            file_name="sales_forecast.csv",
            mime="text/csv",
        )
    with export_col2:
        if st.button("Save trained model"):
            model_path = save_model(df, model_name=selected_model)
            st.success(f"Model saved to `{model_path.name}`")

with performance_tab:
    st.subheader("Hold-out Model Validation")
    st.write(
        "Each model is trained on early months and evaluated on the most recent hold-out period. "
        "MAPE (Mean Absolute Percentage Error) is the primary ranking metric."
    )

    st.dataframe(comparison_df, width="stretch", hide_index=True)
    st.plotly_chart(plot_model_comparison(comparison_df), width="stretch")

    st.subheader("In-sample Linear Regression Fit")
    fit_df = get_actual_vs_predicted(df)
    st.plotly_chart(plot_actual_vs_predicted(fit_df), width="stretch")

st.divider()
footer_col1, footer_col2 = st.columns([3, 1])
with footer_col1:
    st.caption(
        "Portfolio project: sales forecasting with validated ML models and interactive analytics. "
        "Upload a CSV with `Date` and `Sales` columns to try your own data."
    )
with footer_col2:
    st.markdown(
        "[GitHub](https://github.com/Ani0130) · "
        "[LinkedIn](https://www.linkedin.com/in/aniket-jha-1051b4252/)"
    )
