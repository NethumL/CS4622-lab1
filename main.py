# %% [markdown]
# # Lab 1 - 190349K

# %% [markdown]
# ## Info
# 
# ### Requirements
# 
# -   Python libraries (install with `pip install numpy pandas matplotlib seaborn sklearn xgboost`)
#     -   numpy
#     -   pandas
#     -   matplotlib
#     -   seaborn
#     -   sklearn
#     -   xgboost
# -   Datasets
#     -   Should be inside `data/` directory
#     -   `data/train.csv`
#     -   `data/valid.csv`
#     -   `data/test.csv`
# 
# ### Saving and loading models
# 
# -   Models that are trained are also saved to `models/` in the `joblib` format
# -   `models/label_1_before` is the model for label 1 before feature engineering, and `models/label_1_after` is the one after feature engineering
# -   Calls to the `load_model` function have been commented out (running this notebook as is will train each model for the first time)
# -   To reuse the saved models, find the calls to the `save_model` function and comment that line plus the line before it (that trains the model)
# -   Then uncomment the calls to the `load_model` function
# 

# %% [markdown]
# ## Loading and inspecting data

# %%
import pandas as pd
import numpy as np
from sklearn import metrics

# Constants
L1 = 'label_1'
L2 = 'label_2'
L3 = 'label_3'
L4 = 'label_4'
LABELS = [L1, L2, L3, L4]
AGE_LABEL = L2
FEATURES = [f'feature_{i}' for i in range(1, 257)]

# %%
train_df = pd.read_csv("data/train.csv")
train_df.head()

# %%
valid_df = pd.read_csv("data/valid.csv")
valid_df.head()

# %%
test_df = pd.read_csv("data/test.csv")
test_df.head()

# %%
train_df[LABELS + [FEATURES[i] for i in range(0, 256, 32)]].describe()

# %%
train_df.info()

# %% [markdown]
# ## Preprocessing
# 
# - `RobustScaler` is used to scale the features
# - For age (label_2), rows where label is missing are filtered out
# - For accent (label_4), unequal distribution is handled when training the model later on

# %%
from sklearn.preprocessing import RobustScaler

# To store datasets for each label
X_train = {}
X_valid = {}
X_test = {}
y_train = {}
y_valid = {}
y_test = {}
y_pred_before = {}
y_pred_after = {}


def filter_missing_age(df: pd.DataFrame):
    """Filter out rows where age is `NaN`"""
    return df[df[AGE_LABEL].notna()]


# Filter `NaN` and scale datasets
for target_label in LABELS:
    tr_df = filter_missing_age(train_df) if target_label == AGE_LABEL else train_df
    vl_df = filter_missing_age(valid_df) if target_label == AGE_LABEL else valid_df
    ts_df = test_df  # No need to filter rows with missing age in test dataset

    scaler = RobustScaler()
    X_train[target_label] = pd.DataFrame(
        scaler.fit_transform(tr_df.drop(LABELS, axis=1)), columns=FEATURES
    )
    y_train[target_label] = tr_df[target_label]
    X_valid[target_label] = pd.DataFrame(
        scaler.transform(vl_df.drop(LABELS, axis=1)), columns=FEATURES
    )
    y_valid[target_label] = vl_df[target_label]
    X_test[target_label] = pd.DataFrame(
        scaler.transform(ts_df.drop(LABELS, axis=1)), columns=FEATURES
    )
    y_test[target_label] = ts_df[target_label]

# %%
X_train[L1].head()

# %% [markdown]
# ## Training baseline models

# %% [markdown]
# ### Predicting labels and showing statistics

# %%
def filter_nans(y_true: pd.Series, y_pred: pd.Series):
    """Filter `NaN`s in both `y_true` and `y_pred` based on `NaN`s in `y_true`"""
    return y_true[y_true.isna() == False], y_pred[y_true.isna() == False]


def predict(model, X_test: pd.DataFrame, y_test: pd.Series, categorical=True):
    y_pred: pd.Series = model.predict(X_test)
    print("Stats:")
    if categorical:
        print("Confusion matrix:")
        print(metrics.confusion_matrix(y_test, y_pred))
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("Precision:", metrics.precision_score(y_test, y_pred, average="weighted"))
        print("Recall:", metrics.recall_score(y_test, y_pred, average="weighted"))
    else:
        print(
            "RMSE:",
            metrics.mean_squared_error(*filter_nans(y_test, y_pred), squared=False),
        )
    return y_pred

# %% [markdown]
# ### Saving models

# %%
import joblib
import os

MODEL_DIR = "models"


def save_model(model, name: str):
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    joblib.dump(model, f"{MODEL_DIR}/{name}.joblib")


def load_model(name: str):
    return joblib.load(f"{MODEL_DIR}/{name}.joblib")

# %% [markdown]
# ### XGBoost

# %%
import xgboost as xgb


def train_xgboost_binary(X_train: pd.DataFrame, y_train: pd.Series):
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    xgb_model.fit(X_train, y_train)
    return xgb_model


def train_xgboost_regression(X_train: pd.DataFrame, y_train: pd.Series):
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    xgb_model.fit(X_train, y_train)
    return xgb_model

# %% [markdown]
# ### SVM

# %%
from sklearn import svm


def train_svm(X_train: pd.DataFrame, y_train: pd.Series, balance=False, categorical=True):
    if categorical:
        if balance:
            clf = svm.SVC(kernel="linear", class_weight="balanced")
        else:
            clf = svm.SVC(kernel="linear")
    else:
        clf = svm.SVR(kernel='linear')
    clf.fit(X_train, y_train)
    return clf

# %% [markdown]
# ### Training and predicting

# %%
model = train_svm(X_train[L1], y_train[L1])  # To use the pre-saved model, comment out this line and the next one
save_model(model, "label_1_before")
# model = load_model("label_1_before")  # Then uncomment this line to load that pre-saved model
y_pred_before[L1] = predict(model, X_test[L1], y_test[L1])

# %%
model = train_xgboost_regression(X_train[L2], y_train[L2])
save_model(model, "label_2_before")
# model = load_model("label_2_before")
y_pred_before[L2] = predict(model, X_test[L2], y_test[L2], categorical=False)

# %%
model = train_svm(X_train[L3], y_train[L3])
save_model(model, "label_3_before")
# model = load_model("label_3_before")
y_pred_before[L3] = predict(model, X_test[L3], y_test[L3])

# %%
model = train_svm(X_train[L4], y_train[L4], balance=True)
save_model(model, "label_4_before")
# model = load_model("label_4_before")
y_pred_before[L4] = predict(model, X_test[L4], y_test[L4])

# %% [markdown]
# ## Feature engineering
# 
# -   Methods used
#     -   Principal component analysis (PCA)
#     -   Recursive feature elimination (attempted and dropped afterwards)
#     -   Univariate feature selection
# 

# %%
from enum import Enum
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, f_classif, r_regression, SelectFromModel
from sklearn.ensemble import RandomForestClassifier


class Model(Enum):
    SVC = "SVC"
    XGB = "XGBoost regressor"
    RANDOM_FOREST = "Random forest"
    LINEAR = "Linear regressor"


def fit_and_transform_pca(X_train: pd.DataFrame, X_test: pd.DataFrame):
    pca = PCA(n_components=0.95, svd_solver="full")
    pca.fit(X_train)
    X_train_trf = pd.DataFrame(pca.transform(X_train))
    X_test_trf = pd.DataFrame(pca.transform(X_test))
    print("Shape after PCA:", X_train_trf.shape)
    return X_train_trf, X_test_trf


# Not used as it is time-consuming
def transform_with_rfe(X: pd.DataFrame, y: pd.Series):
    rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
    rfe.fit(X, y)
    print("Shape after RFE:", X.shape)
    return rfe, rfe.transform(X)


def univariate_feature_selection(
    X: pd.DataFrame, y: pd.Series, categorical=True, feature_count=30
):
    if categorical:
        score_func = f_classif
    else:
        score_func = r_regression
    selector = SelectKBest(score_func, k=feature_count)
    X_new = selector.fit_transform(X, y)
    print("Shape after univariate:", X_new.shape)
    return selector, X_new

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from typing import Dict

# Test datasets after transforming for each label
X_test_transformed: Dict[str, pd.DataFrame] = {}


def transform_train_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_label: str,
    categorical=True,
    model_type: Model = Model.SVC,
    feature_count=30,
    pca_count=5
):
    X_train_trf, X_test_trf = fit_and_transform_pca(X_train, X_test)

    # # Recursive feature elimination is commented out as it is very time consuming
    # rfe, X_train_trf = transform_with_rfe(X_train, y_train)
    # X_test_trf = rfe.transform(X_test_trf)

    # Skip univariate feature selection if `feature_count` is specified as 0
    if feature_count != 0:
        selector, X_train_trf = univariate_feature_selection(
            X_train_trf, y_train, categorical=categorical, feature_count=feature_count
        )
        X_test_trf = pd.DataFrame(selector.transform(X_test_trf))
    
    # Re-run PCA multiple times
    for _ in range(pca_count):
        X_train_trf, X_test_trf = fit_and_transform_pca(X_train_trf, X_test_trf)
    # X_train_trf, X_test_trf = fit_and_transform_pca(X_train_trf, X_test_trf)
    print("Model:", model_type)

    # Training model
    if categorical or model_type == Model.SVC:
        model = svm.SVC(kernel="rbf", class_weight="balanced")
    elif model_type == Model.XGB:
        model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    elif model_type == Model.RANDOM_FOREST:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == Model.LINEAR:
        model = LinearRegression()
    model.fit(X_train_trf, y_train)
    save_model(model, f"{target_label}_after")

    # model = load_model(f"{target_label}_after")
    y_pred = predict(model, X_test_trf, y_test, categorical=categorical)
    return y_pred, X_test_trf

# %% [markdown]
# ### Training and predicting

# %%
y_pred_after[L1], X_test_transformed[L1] = transform_train_predict(
    X_train[L1], y_train[L1], X_test[L1], y_test[L1], L1, feature_count=0, pca_count=5
)

# %%
y_pred_after[L2], X_test_transformed[L2] = transform_train_predict(
    X_train[L2],
    y_train[L2],
    X_test[L2],
    y_test[L2],
    L2,
    categorical=False,
    model_type=Model.XGB,
    feature_count=0,
    pca_count=6
)

# %%
y_pred_after[L3], X_test_transformed[L3] = transform_train_predict(
    X_train[L3], y_train[L3], X_test[L3], y_test[L3], L3, feature_count=0, pca_count=1
)

# %%
y_pred_after[L4], X_test_transformed[L4] = transform_train_predict(
    X_train[L4], y_train[L4], X_test[L4], y_test[L4], L4, feature_count=0, pca_count=2
)

# %%
OUT_COLS_FIRST = [
    "Predicted labels before feature engineering",
    "Predicted labels after feature engineering",
    "No of new features",
]
OUT_COLS_FEATURES = [f"new_feature_{i}" for i in range(1, 257)]
OUT_COLS = OUT_COLS_FIRST + OUT_COLS_FEATURES


def save_results_to_csv(label: str, no_of_features: int, X_test: pd.DataFrame):
    df = pd.DataFrame([], columns=OUT_COLS)
    df[OUT_COLS[0]] = y_pred_before[label]
    df[OUT_COLS[1]] = y_pred_after[label]
    col2 = np.empty(len(y_pred_before[label]))
    col2.fill(no_of_features)
    df[OUT_COLS[2]] = col2.astype(int)
    for i in range(1, no_of_features+1):
        df[f"new_feature_{i}"] = X_test[i - 1]
    empty_col = np.empty(len(y_pred_before[label]))
    empty_col.fill(0)
    for i in range(no_of_features+1, 257):
        df[f"new_feature_{i}"] = empty_col.astype(int)
    df.to_csv(f"results/190349K_{label}.csv", index=False)


save_results_to_csv(L1, len(X_test_transformed[L1].columns), X_test_transformed[L1])
save_results_to_csv(L2, len(X_test_transformed[L2].columns), X_test_transformed[L2])
save_results_to_csv(L3, len(X_test_transformed[L3].columns), X_test_transformed[L3])
save_results_to_csv(L4, len(X_test_transformed[L4].columns), X_test_transformed[L4])

# %% [markdown]
# ### Plot correlation between features
# 
# - This shows that there is no longer any significant correlation between the engineered features

# %%
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

def plot_correlation(target_label: str):
    correlation_matrix = X_test_transformed[target_label].corr()
    correlation_threshold = 0.5

    filtered_correlation_matrix = correlation_matrix[
        (correlation_matrix > correlation_threshold) | (correlation_matrix < -correlation_threshold)
    ]
    plt.figure(figsize=(10, 8))
    sns.heatmap(filtered_correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(f"Correlation Heatmap (Filtered) for {target_label.replace('_', ' ')}")
    plt.savefig(f"plots/correlation_{target_label}.pdf")

# %%
plot_correlation(L1)

# %%
plot_correlation(L2)

# %%
plot_correlation(L3)

# %%
plot_correlation(L4)


