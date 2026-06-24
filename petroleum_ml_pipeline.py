"""
Petroleum Oil Recovery Prediction with SHAP Explainability
Dataset: Equinor Volve North Sea field (open access)

Predicts daily oil production (BORE_OIL_VOL) from well operating
parameters using five regression models, then applies SHAP analysis
to the best model to identify which reservoir parameters actually
drive the prediction.

Run order: load_data -> clean_raw_data -> fix_gauge_and_engineer_features
-> build_model_dataset -> split_data -> train_baseline_models
-> tune_xgboost -> run_shap_analysis -> save_artifacts
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor


DATA_PATH = "Volve production data-Daily Production Data.csv"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------

def load_data(path):
    """Load the raw Volve CSV and fix column types.

    The volume columns load as text because the source file mixes
    numeric and blank string values, so they need an explicit
    numeric conversion.
    """
    df = pd.read_csv(path)

    numeric_cols = ["BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL", "BORE_WI_VOL"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["DATEPRD"] = pd.to_datetime(df["DATEPRD"], format="%d-%b-%y", errors="coerce")
    df["DATEPRD"] = df["DATEPRD"].fillna(pd.to_datetime(df["DATEPRD"], errors="coerce"))
    return df


# ---------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------

def clean_raw_data(df):
    """Remove rows and columns that don't belong in the modeling set.

    Injector wells are dropped because they don't produce oil at all,
    shut-in days are dropped because the well wasn't flowing, and a
    handful of columns are dropped for being either administrative,
    a target leakage risk, or redundant with another column.
    """
    df = df[df["WELL_TYPE"] == "OP"].copy()

    # water volume can't physically be negative, this is a sensor artifact
    df["BORE_WAT_VOL"] = df["BORE_WAT_VOL"].clip(lower=0)

    # drop shut-in days, then fix a reporting artifact where a few rows show 25 hours
    df = df[df["ON_STREAM_HRS"] > 0].copy()
    df["ON_STREAM_HRS"] = df["ON_STREAM_HRS"].clip(upper=24)

    drop_cols = [
        "NPD_WELL_BORE_CODE", "NPD_WELL_BORE_NAME", "NPD_FIELD_CODE",
        "NPD_FIELD_NAME", "NPD_FACILITY_CODE", "NPD_FACILITY_NAME",
        "AVG_CHOKE_UOM",            # administrative, no predictive value
        "FLOW_KIND", "WELL_TYPE",   # constant once filtered to producers only
        "BORE_WI_VOL",              # not applicable once injectors are removed
        "BORE_GAS_VOL",             # co-produced with oil at the same wellhead event, causes leakage
        "AVG_DOWNHOLE_TEMPERATURE", "AVG_DP_TUBING",  # correlate >0.9 with downhole pressure
        "DP_CHOKE_SIZE",            # correlates >0.9 with wellhead pressure
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


def fix_gauge_and_engineer_features(df):
    """Fix the downhole pressure gauge failure and add PRESSURE_DRAWDOWN.

    Well F-12H and F-5AH have stretches where the downhole pressure
    gauge failed and recorded near-zero instead of a true reading.
    Treating those as missing and filling per well (with a global
    median fallback for F-5AH, whose entire history is invalid)
    avoids a fabricated zero distorting the drawdown calculation.
    """
    df["DOWNHOLE_GAUGE_VALID"] = (df["AVG_DOWNHOLE_PRESSURE"] >= 10).astype(int)

    df = df.sort_values(["WELL_BORE_CODE", "DATEPRD"])
    df.loc[df["DOWNHOLE_GAUGE_VALID"] == 0, "AVG_DOWNHOLE_PRESSURE"] = np.nan

    df["AVG_DOWNHOLE_PRESSURE"] = (
        df.groupby("WELL_BORE_CODE")["AVG_DOWNHOLE_PRESSURE"]
        .transform(lambda x: x.ffill().bfill())
    )
    df["AVG_DOWNHOLE_PRESSURE"] = df["AVG_DOWNHOLE_PRESSURE"].fillna(
        df["AVG_DOWNHOLE_PRESSURE"].median()
    )

    for col in ["AVG_ANNULUS_PRESS", "AVG_WHP_P", "AVG_WHT_P", "BORE_WAT_VOL"]:
        df[col] = df.groupby("WELL_BORE_CODE")[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

    # drawdown, not absolute pressure, is what actually drives flow
    df["PRESSURE_DRAWDOWN"] = df["AVG_DOWNHOLE_PRESSURE"] - df["AVG_WHP_P"]

    # cap extreme water volume readings instead of deleting the row,
    # since the same row's oil volume is still valid and useful
    q1, q3 = df["BORE_WAT_VOL"].quantile([0.25, 0.75])
    upper_bound = q3 + 1.5 * (q3 - q1)
    df["BORE_WAT_VOL"] = df["BORE_WAT_VOL"].clip(upper=upper_bound)

    return df


def build_model_dataset(df):
    """Drop rows with no target, encode well identity, and finalize X/y.

    BORE_OIL_VOL is never imputed; gaps reflect Volve's periodic
    well-test reporting and rows without it are excluded.

    AVG_DOWNHOLE_PRESSURE is dropped because it correlates 0.92 with
    PRESSURE_DRAWDOWN once the gauge fix is applied.
    """
    df = df.dropna(subset=["BORE_OIL_VOL"]).copy()
    df = pd.get_dummies(df, columns=["WELL_BORE_CODE"], prefix="WELL")

    drop_cols = ["DATEPRD",'BORE_WAT_VOL', "AVG_DOWNHOLE_PRESSURE", "is_producing", "BORE_OIL_VOL"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["BORE_OIL_VOL"]
    return X, y


def split_data(X, y):
    """Random 80/20 split, the methodology used for every benchmark in the paper.

    A chronological split was tried and rejected: one well (F-5AH) falls
    entirely inside the held-out date range and never appears in training,
    and the field's natural production decline over its 8-year life means
    train and test come from different distributions either way. A random
    split keeps every well represented in both sets and matches the
    methodology already reported for all five models.
    """
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


# ---------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------

def make_pipeline(model):
    """Bundle a regressor with MinMax scaling.

    Wrapping each model this way means raw, unscaled features can be
    passed in directly, and a model fit inside cross-validation gets
    its scaler refit on that fold's training data only, instead of
    leaking scale information from the held-out fold.
    """
    return Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", model),
    ])


def evaluate(name, y_true, y_pred):
    return {
        "Model": name,
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def train_baseline_models(X_train, X_test, y_train, y_test):
    candidates = {
        "Linear Regression": LinearRegression(),
        "SVR": SVR(kernel="rbf", C=100, gamma="scale"),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "ANN": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                             random_state=RANDOM_STATE, early_stopping=True),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.08, gamma=0, subsample=0.75,
                                 colsample_bytree=1, max_depth=7, random_state=RANDOM_STATE),
    }

    results = []
    pipelines = {}
    for name, model in candidates.items():
        pipe = make_pipeline(model)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append(evaluate(name, y_test, pred))
        pipelines[name] = pipe

    return pd.DataFrame(results), pipelines


def tune_xgboost(X_train, y_train, X_test, y_test):
    """Grid search XGBoost hyperparameters inside the same scaling pipeline."""
    pipe = make_pipeline(XGBRegressor(random_state=RANDOM_STATE))

    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.8, 1.0],
    }

    grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring="r2", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_pipe = grid_search.best_estimator_
    pred = best_pipe.predict(X_test)
    result = evaluate("XGBoost (tuned)", y_test, pred)

    print("Best XGBoost params:", grid_search.best_params_)
    return best_pipe, result



# ---------------------------------------------------------------
# Figures
# ---------------------------------------------------------------

def plot_correlation_heatmap(X_train_scaled, y_train, path):
    fig_df = X_train_scaled.copy()
    fig_df["BORE_OIL_VOL"] = y_train.values
    corr = fig_df.corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", mask=mask,
                vmin=-1, vmax=1, linewidths=0.5, annot_kws={"size": 9})
    plt.title("Figure 1: Correlation Heatmap, Training Set")
    plt.tight_layout()
    print("\nFigure1: Correlation Heatmap")
    plt.show()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_model_comparison(results_df, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].bar(results_df["Model"], results_df["R2"], color="#2E5984", edgecolor="black")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("R2 Score")
    axes[0].set_title("Model Comparison, R2")
    axes[0].tick_params(axis="x", rotation=20)

    x = np.arange(len(results_df))
    axes[1].bar(x - 0.175, results_df["RMSE"], 0.35, label="RMSE", color="#BF616A", edgecolor="black")
    axes[1].bar(x + 0.175, results_df["MAE"], 0.35, label="MAE", color="#D08770", edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(results_df["Model"], rotation=20)
    axes[1].set_ylabel("Error (barrels)")
    axes[1].set_title("Model Comparison, RMSE and MAE")
    axes[1].legend()

    plt.tight_layout()
    print("\nFigure2: Model Comparison Chart")
    plt.show()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_predicted_vs_actual(y_test, y_pred, path):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.4, s=20, color="steelblue", edgecolor="k", linewidth=0.3)
    lims = [0, max(y_test.max(), y_pred.max()) * 1.05]
    plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Prediction")
    plt.xlabel("Actual BORE_OIL_VOL")
    plt.ylabel("Predicted BORE_OIL_VOL")
    plt.title("Figure 2: Predicted vs Actual, Tuned XGBoost")
    plt.legend()
    plt.tight_layout()
    print("\nFigure2: Model Treadline")
    plt.show()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def run_shap_analysis(best_pipe, X_test, output_dir):
    """Run SHAP on the underlying model, after scaling X_test the same way the pipeline does.

    TreeExplainer is used because it computes exact Shapley values for
    tree models directly, instead of the slower sampling-based approach
    a model-agnostic explainer would need.
    """
    scaler = best_pipe.named_steps["scaler"]
    model = best_pipe.named_steps["model"]
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)

    shap.summary_plot(shap_values, X_test_scaled, show=False)
    print("\nFigure3: SHAP summary")
    plt.savefig(f"{output_dir}/figure3_shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    explanation = explainer(X_test_scaled)
    shap.plots.waterfall(explanation[0], max_display=10, show=False)
    plt.tight_layout()
    print("\nFigure4: SHAP Waterfall")
    plt.savefig(f"{output_dir}/figure4_shap_waterfall.png", dpi=200, bbox_inches="tight")
    plt.close()

    shap.dependence_plot("PRESSURE_DRAWDOWN", shap_values, X_test_scaled, show=False)
    print("\nFigure5: SHAP Dependence Drawdown")
    plt.show()
    plt.savefig(f"{output_dir}/figure5_shap_dependence_drawdown.png", dpi=200, bbox_inches="tight")
    plt.close()

    importance_ranking = pd.Series(
        np.abs(shap_values).mean(axis=0), index=X_test.columns
    ).sort_values(ascending=False)

    return importance_ranking



# ---------------------------------------------------------------
# Saving for deployment
# ---------------------------------------------------------------

def save_artifacts(best_pipe, feature_columns, output_dir):
    """Save the three files app.py loads, so the repo is reproducible
    end-to-end from this script alone.
    """
    with open(f"{output_dir}/model.pkl", "wb") as f:
        pickle.dump(best_pipe.named_steps["model"], f)
    with open(f"{output_dir}/scaler.pkl", "wb") as f:
        pickle.dump(best_pipe.named_steps["scaler"], f)
    with open(f"{output_dir}/feature_columns.pkl", "wb") as f:
        pickle.dump(list(feature_columns), f)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    df = load_data(DATA_PATH)
    df = clean_raw_data(df)
    df = fix_gauge_and_engineer_features(df)

    X, y = build_model_dataset(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    baseline_results, baseline_pipelines = train_baseline_models(X_train, X_test, y_train, y_test)
    best_pipe, xgb_result = tune_xgboost(X_train, y_train, X_test, y_test)

    all_results = pd.concat([baseline_results, pd.DataFrame([xgb_result])], ignore_index=True)
    print(all_results)

    train_scaled = pd.DataFrame(
        best_pipe.named_steps["scaler"].transform(X_train), columns=X_train.columns, index=X_train.index
    )
    plot_correlation_heatmap(train_scaled, y_train, f"{OUTPUT_DIR}/figure1_correlation_heatmap.png")
    plot_model_comparison(all_results, f"{OUTPUT_DIR}/model_comparison_chart.png")
    plot_predicted_vs_actual(y_test, best_pipe.predict(X_test), f"{OUTPUT_DIR}/figure2_predicted_vs_actual.png")

    importance_ranking = run_shap_analysis(best_pipe, X_test, OUTPUT_DIR)
    print("\nSHAP feature importance ranking:")
    print(importance_ranking)

    save_artifacts(best_pipe, X.columns, OUTPUT_DIR)
    print(f"\nDone. Figures and model artifacts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
