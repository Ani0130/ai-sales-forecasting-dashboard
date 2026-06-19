from io import BytesIO
from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"
REQUIRED_COLUMNS = {"Date", "Sales"}


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Date", "Sales"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def validate_sales_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    errors = []

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(sorted(missing_columns))}")
        return df, errors

    prepared = _prepare_dataframe(df)

    if prepared.empty:
        errors.append("No valid rows found after parsing Date and Sales.")
        return prepared, errors

    if len(prepared) < 6:
        errors.append("At least 6 months of sales data are required for reliable forecasting.")

    if (prepared["Sales"] < 0).any():
        errors.append("Sales values must be zero or positive.")

    if prepared["Date"].duplicated().any():
        errors.append("Duplicate dates found. Each month should appear only once.")

    return prepared, errors


def load_data(path: Path | str | None = None) -> pd.DataFrame:
    data_path = Path(path) if path else DEFAULT_DATA_PATH
    df = pd.read_csv(data_path)
    prepared, errors = validate_sales_data(df)

    if errors:
        raise ValueError("; ".join(errors))

    return prepared


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.read()
    df = pd.read_csv(BytesIO(content))
    prepared, errors = validate_sales_data(df)

    if errors:
        raise ValueError("; ".join(errors))

    return prepared
