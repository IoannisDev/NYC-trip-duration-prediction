# NYC Taxi Duration

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Commands

### 1. Process data

Place the raw dataset at `data/raw/NYC.csv`, then run:

```bash
python src/data/make_dataset.py
```

### 2. Train the model

```bash
python src/model/train_model.py
```

### 3. Predict

**Batch** — pass a CSV file and get predictions saved to a CSV:

```bash
python src/model/predict_model.py --input <path/to/input.csv> --output <path/to/output.csv>
```

**Interactive** — enter a single trip's details at the prompt:

```bash
python src/model/predict_model.py
```

**Optional flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--input`, `-i` | Input CSV for batch prediction | — |
| `--output`, `-o` | Output CSV path | `results/predictions.csv` |
| `--model`, `-m` | Path to trained model `.pkl` | `models/hist_model.pkl` |
| `--geojson` | Path to NYC boroughs GeoJSON | `data/processed/boundary.geojson` |
