from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from pandas import DataFrame
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import joblib
import os
import sys
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open(os.path.join(BASE_DIR, "config.yml"), "r") as f:
    _config = yaml.safe_load(f)

TRAIN_DATA_PATH   = os.path.join(BASE_DIR, _config["data"]["processed_data_path"])
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, _config["model"]["output_path"])

HIST_PARAM = {'learning_rate': 0.20822254137715343,
               'l2_regularization': 1.5234018422544593,
               'max_leaf_nodes': 100,
               'max_depth': 10,
               'min_samples_leaf': 33}

def load_data(path) -> DataFrame:
    df = pd.read_csv(path)
    return df

def prepare_xy(df: DataFrame) -> tuple:
    X = df.drop(['trip_duration', 'id', 'pickup_datetime', 'index_left'],errors='ignore', axis=1)
    y = df['trip_duration']
    categoricals = X.select_dtypes(include=['object', 'str']).columns.tolist()
    X[categoricals] = X[categoricals].astype('category')
    return X, y, categoricals

def train(df: DataFrame) -> tuple:
    X, y, categorical = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training the model...")
    model = HistGradientBoostingRegressor(**HIST_PARAM, categorical_features=categorical, random_state=42)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    mae = mean_absolute_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)
    print(f"Baseline Mean Absolute Error: {mae:.2f}")
    print(f"R2: {r2:.2f}")
    return model, mae, r2

def main():
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"Processed data not found at: {TRAIN_DATA_PATH}")
        print("Please run: python src/data/make_dataset.py")
        sys.exit(1)

    df = load_data(TRAIN_DATA_PATH)
    model, mae, r2 = train(df)

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
