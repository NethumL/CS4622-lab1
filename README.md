# CS4622 Lab 1

## Setup

- Install python dependencies

```sh
pip install -r requirements.txt
```

- Datasets
  - Should be inside `data/` directory
  - `data/train.csv`
  - `data/valid.csv`
  - `data/test.csv`

## How it works

### Saving and loading models

- Models that are trained are also saved to `models/` in the `joblib` format
- `models/label_1_before` is the model for label 1 before feature engineering, and `models/label_1_after` is the one after feature engineering
- Calls to the `load_model` function have been commented out (running this notebook as is will train each model for the first time)
- To reuse the saved models, find the calls to the `save_model` function and comment that line plus the line before it (that trains the model)
- Then uncomment the calls to the `load_model` function

### Preprocessing

- `RobustScaler` is used to scale the features
- For age (label_2), rows where label is missing are filtered out
- For accent (label_4), unequal distribution is handled when training the model later on

### Feature engineering

- Methods used
  - Principal component analysis (PCA)
  - Recursive feature elimination (attempted and dropped afterwards)
  - Univariate feature selection
