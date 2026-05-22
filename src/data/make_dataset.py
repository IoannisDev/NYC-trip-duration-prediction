import pandas as pd
import yaml
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.preprocess.build_features import make_feat

def main():
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    raw_path = config["data"]["raw_data_path"]
    processed_path = config["data"]["processed_data_path"]

    print(f"Loading data from: {raw_path}")
    df = pd.read_csv(raw_path)
    nulls = df.isnull().sum()
    if nulls.any():
        print(" Nulls found in raw data:")
        print(nulls[nulls > 0])
    
    print("Building features and cleaning data...")
    df = make_feat(df)
    null_counts = df.isnull().sum()
    if null_counts.any():
        print("Nulls found after feature building:")
        print(null_counts[null_counts > 0])

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Saved cleaned data to: {processed_path}")

if __name__ == "__main__":
    main()
