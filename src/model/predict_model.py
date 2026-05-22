import argparse
import os
import sys
import numpy as np
import pandas as pd
import joblib
import geopandas as gp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.preprocess.build_features import haversine_form

DEFAULT_MODEL_PATH   = os.path.join(BASE_DIR, "models", "hist_model.pkl")
DEFAULT_GEOJSON_PATH = os.path.join(BASE_DIR, "data", "processed", "boundary.geojson")
DEFAULT_OUTPUT_PATH  = os.path.join(BASE_DIR, "results", "predictions.csv")

FEATURE_COLUMNS = [
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "distance",
    "day_of_week",
    "pickup_hour",
    "is_rush_hour",
    "is_weekend",
    "BoroName",
]


def build_features(df: pd.DataFrame, geojson_path: str) -> pd.DataFrame:
    df = df.copy()

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["day_of_week"]  = df["pickup_datetime"].dt.day_of_week
    df["pickup_hour"]  = df["pickup_datetime"].dt.hour
    df["is_rush_hour"] = df["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype(int)

    df["distance"] = haversine_form(
        df["pickup_latitude"],  df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )

    if not os.path.exists(geojson_path):
        print(f"[WARNING] GeoJSON not found at {geojson_path}. 'BoroName' will be NaN.")
        df["BoroName"] = np.nan
    else:
        boroughs = gp.read_file(geojson_path)
        boroughs["geometry"] = boroughs["geometry"].buffer(0)
        boroughs = boroughs.to_crs("EPSG:4326")

        gdf = gp.GeoDataFrame(
            df,
            geometry=gp.points_from_xy(df["dropoff_longitude"], df["dropoff_latitude"]),
            crs="EPSG:4326",
        ).rename_geometry("points_geom")

        result = gp.sjoin(gdf, boroughs[["BoroName", "geometry"]], how="left", predicate="within")
        df = pd.DataFrame(result.drop(columns=["points_geom", "geometry", "index_right"], errors="ignore"))

    return df


def prepare_for_model(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns after feature engineering: {missing}")

    X = df[FEATURE_COLUMNS].copy()

    cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
    if "BoroName" in X.columns and "BoroName" not in cat_cols:
        cat_cols.append("BoroName")
    X[cat_cols] = X[cat_cols].astype("category")

    return X


def load_model(model_path: str):
    if not os.path.exists(model_path):
        sys.exit(
            f"[ERROR] Model file not found: {model_path}\n"
            "Run  python src/model/train_model.py  first."
        )
    print(f"Loading model from: {model_path}")
    return joblib.load(model_path)


def predict_batch(
    input_path: str,
    output_path: str,
    model_path: str  = DEFAULT_MODEL_PATH,
    geojson_path: str = DEFAULT_GEOJSON_PATH,
) -> pd.DataFrame:
    if not os.path.exists(input_path):
        sys.exit(f"[ERROR] Input file not found: {input_path}")

    print(f"Loading input data from: {input_path}")
    df_raw = pd.read_csv(input_path)
    print(f"  Rows: {len(df_raw):,}")

    print("Engineering features…")
    df_feat = build_features(df_raw, geojson_path)

    model = load_model(model_path)

    X = prepare_for_model(df_feat)
    print(f"Running predictions on {len(X):,} rows…")
    preds = model.predict(X)

    df_raw = df_raw.iloc[: len(preds)].copy()
    df_raw["predicted_duration_min"] = np.round(preds, 2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_raw.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

    print("\n--- Prediction Summary ---")
    print(f"  Count : {len(preds):,}")
    print(f"  Mean  : {preds.mean():.2f} min")
    print(f"  Median: {np.median(preds):.2f} min")
    print(f"  Min   : {preds.min():.2f} min")
    print(f"  Max   : {preds.max():.2f} min")

    return df_raw


def _prompt_float(label: str, default=None) -> float:
    hint = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"  {label}{hint}: ").strip()
        if raw == "" and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("    ✗ Please enter a numeric value.")


def _prompt_datetime(label: str) -> str:
    fmt = "%Y-%m-%d %H:%M:%S"
    while True:
        raw = input(f"  {label} (YYYY-MM-DD HH:MM:SS): ").strip()
        try:
            pd.to_datetime(raw)
            return raw
        except Exception:
            print("    ✗ Invalid datetime. Example: 2016-06-15 08:30:00")


def predict_single(
    model_path:   str = DEFAULT_MODEL_PATH,
    geojson_path: str = DEFAULT_GEOJSON_PATH,
) -> float:
    print("\nNYC Taxi Duration — Single Trip Prediction \n")
    print("Enter trip details (press Enter to keep defaults where shown):\n")

    row = {
        "pickup_datetime":   _prompt_datetime("Pickup datetime"),
        "pickup_latitude":   _prompt_float("Pickup latitude",  40.7580),
        "pickup_longitude":  _prompt_float("Pickup longitude", -73.9855),
        "dropoff_latitude":  _prompt_float("Dropoff latitude",  40.6892),
        "dropoff_longitude": _prompt_float("Dropoff longitude", -74.0445),
        "passenger_count":   int(_prompt_float("Passenger count", 1)),
    }

    df_raw  = pd.DataFrame([row])
    df_feat = build_features(df_raw, geojson_path)

    model = load_model(model_path)
    X     = prepare_for_model(df_feat)
    pred  = model.predict(X)[0]

    print(f"\n Predicted trip duration: {pred:.1f} minutes")
    return pred


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict NYC taxi trip duration using a trained model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",  "-i",
        help="Path to input CSV for batch prediction. Omit for interactive single prediction.",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path for output CSV with predictions (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to trained model .pkl file (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--geojson",
        default=DEFAULT_GEOJSON_PATH,
        help=f"Path to NYC boroughs GeoJSON (default: {DEFAULT_GEOJSON_PATH}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.input:
        predict_batch(
            input_path   = args.input,
            output_path  = args.output,
            model_path   = args.model,
            geojson_path = args.geojson,
        )
    else:
        predict_single(
            model_path   = args.model,
            geojson_path = args.geojson,
        )


if __name__ == "__main__":
    main()
