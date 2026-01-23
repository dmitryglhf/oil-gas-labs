import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    # Core imports
    import random as rd
    import warnings
    from pathlib import Path

    import lasio
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    warnings.filterwarnings("ignore")

    # ML
    import lightgbm as lgb
    import optuna
    from optuna.samplers import TPESampler
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import GroupKFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Resolve root directory
    ROOT_DIR = Path(__file__).resolve().parent.parent
    return (
        GroupKFold,
        Path,
        ROOT_DIR,
        TPESampler,
        accuracy_score,
        classification_report,
        f1_score,
        lasio,
        lgb,
        np,
        optuna,
        pd,
        rd,
        tqdm,
    )


@app.cell
def _(ROOT_DIR, lasio, pd, tqdm):
    # Load training data
    dfs = []
    data_path = ROOT_DIR / "task2" / "example" / "data"
    train_test_path = data_path / "train_test"

    for file in tqdm(list(train_test_path.glob("*.las"))):
        try:
            las_file = lasio.read(str(file)).df().reset_index()
            las_file["Well"] = file.stem
            dfs.append(las_file)
        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    train_data = pd.concat(dfs, ignore_index=True)
    train_data.columns = ["DEPT", "SP", "GR", "DT", "DENS", "LITHO", "Well"]

    train_data["LITHO"] = train_data["LITHO"].astype(int)
    print(f"Loaded {len(train_data)} samples from {train_data['Well'].nunique()} wells")
    return data_path, train_data


@app.cell
def _(np, pd):
    # --- Feature Engineering ---
    def make_features(df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()

        # 1. Physics & Ratios
        df["GR_DENS_ratio"] = df["GR"] / (df["DENS"] + 1e-6)
        df["SP_GR_ratio"] = df["SP"] / (df["GR"] + 1e-6)
        df["DT_DENS_ratio"] = df["DT"] / (df["DENS"] + 1e-6)
        df["Porosity_proxy"] = np.exp(-df["DENS"])

        base_cols = ["SP", "GR", "DT", "DENS"]

        # 2. Multi-scale Context (Rolling features)
        # 5 = ~0.5m (локальные изменения)
        # 30 = ~3.0m (свиты/пачки)
        windows = [5, 30]

        for col in base_cols:
            # Gradients
            df[f"{col}_diff1"] = df.groupby("Well")[col].diff().fillna(0)
            df[f"{col}_diff2"] = df.groupby("Well")[col].diff().diff().fillna(0)

            # Rolling stats
            grouped = df.groupby("Well")[col]
            for w in windows:
                mean_col = grouped.transform(
                    lambda x: x.rolling(w, center=True, min_periods=1).mean()
                )
                std_col = grouped.transform(
                    lambda x: x.rolling(w, center=True, min_periods=1).std()
                )

                df[f"{col}_mean_{w}"] = mean_col
                df[f"{col}_std_{w}"] = std_col.fillna(0)

                # Z-score (аномальность)
                df[f"{col}_z_{w}"] = (df[col] - mean_col) / (std_col + 1e-6)

                # Min/Max range
                df[f"{col}_max_{w}"] = grouped.transform(
                    lambda x: x.rolling(w, center=True, min_periods=1).max()
                )
                df[f"{col}_min_{w}"] = grouped.transform(
                    lambda x: x.rolling(w, center=True, min_periods=1).min()
                )

        df = df.fillna(0)

        # 3. Normalization (Robust)
        for col in base_cols:
            df[f"{col}_norm"] = df.groupby("Well")[col].transform(
                lambda x: (x - x.median())
                / (x.quantile(0.75) - x.quantile(0.25) + 1e-6)
            )

        df["GR_log"] = np.log1p(df["GR"].clip(lower=0))

        return df
    return (make_features,)


@app.cell
def _(make_features, train_data):
    df = make_features(train_data)

    # Feature Selection
    ignore_cols = ["Well", "LITHO", "DEPT", "index", "level_0"]
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    print(f"Generated {len(feature_cols)} features.")
    return df, feature_cols


@app.cell
def _(df, feature_cols, rd):
    # Split Train/Test
    train_part_size = 0.7
    rd.seed(17)

    all_wells = df.Well.unique().tolist()
    train_wells = rd.sample(all_wells, round(len(all_wells) * train_part_size))

    train_set = df.loc[df.Well.isin(train_wells)].reset_index(drop=True)
    test_set = df.loc[~df.Well.isin(train_wells)].reset_index(drop=True)

    X_train = train_set[feature_cols]
    y_train = train_set["LITHO"].astype(int)
    g_train = train_set["Well"]

    X_test = test_set[feature_cols]
    y_test = test_set["LITHO"].astype(int)

    print(f"Train samples: {len(X_train)} ({len(train_wells)} wells)")
    print(f"Test samples: {len(X_test)}")
    return X_test, X_train, g_train, y_test, y_train


@app.cell
def _(GroupKFold, TPESampler, f1_score, lgb, np, optuna):
    # --- Single Model Optimizer ---

    def tune_lgb(X, y, groups, n_trials=30, seed=42):
        cv = GroupKFold(n_splits=3)

        print("=" * 50)
        print(f"Tuning LightGBM ({n_trials} trials)")
        print("=" * 50)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.1, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "max_depth": trial.suggest_int("max_depth", 5, 15),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "random_state": seed,
                "n_jobs": -1,
                "verbosity": -1,
                "metric": "binary_logloss",
            }

            # CV Evaluation
            scores = []
            for t, v in cv.split(X, y, groups):
                m = lgb.LGBMClassifier(**params)
                m.fit(X.iloc[t], y.iloc[t])

                # Predict Probabilities to find best threshold later
                probs = m.predict_proba(X.iloc[v])[:, 1]

                # Simple threshold check for objective
                pred = (probs >= 0.5).astype(int)
                scores.append(f1_score(y.iloc[v], pred))

            return np.mean(scores)

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_params.update({"random_state": seed, "n_jobs": -1, "verbosity": -1})
        print(f"Best CV F1 (approx): {study.best_value:.4f}")

        # --- Find Exact Best Threshold on OOF ---
        print("\nCalculating Best Threshold on OOF...")
        oof_preds = np.zeros(len(X))
        for t, v in cv.split(X, y, groups):
            m = lgb.LGBMClassifier(**best_params)
            m.fit(X.iloc[t], y.iloc[t])
            oof_preds[v] = m.predict_proba(X.iloc[v])[:, 1]

        best_thr = 0.5
        best_f1 = 0
        for thr in np.linspace(0.3, 0.7, 100):
            score = f1_score(y, (oof_preds >= thr).astype(int))
            if score > best_f1:
                best_f1 = score
                best_thr = thr

        print(f"Optimal Threshold: {best_thr:.3f} (OOF F1: {best_f1:.4f})")

        return best_params, best_thr
    return (tune_lgb,)


@app.cell
def _(X_train, g_train, tune_lgb, y_train):
    # Run Tuning
    best_params, best_thr = tune_lgb(X_train, y_train, g_train, n_trials=40)
    return best_params, best_thr


@app.cell
def _(
    X_test,
    X_train,
    accuracy_score,
    best_params,
    best_thr,
    classification_report,
    f1_score,
    lgb,
    y_test,
    y_train,
):
    # Validate Single Best Model
    print("Training Final LGBM on full train set...")
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)

    # Predict
    probs = model.predict_proba(X_test)[:, 1]

    # Raw Thresholding ONLY (Smoothing removed)
    pred_final = (probs >= best_thr).astype(int)

    print("\n" + "=" * 40)
    print("TEST SET RESULTS")
    print("=" * 40)
    print(f"F1 Score:         {f1_score(y_test, pred_final):.4f}")
    print(f"Accuracy:         {accuracy_score(y_test, pred_final):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, pred_final))
    return


@app.cell
def _(data_path, feature_cols, make_features, pd):
    # Load Validation Data
    validation_data = pd.read_csv(data_path / "Shestakovo_validation.csv")
    val_df_full = make_features(validation_data)

    X_valid = val_df_full[feature_cols]

    print(f"Validation shape: {X_valid.shape}")
    return (X_valid,)


@app.cell
def _(X_valid, best_params, best_thr, df, feature_cols, lgb):
    # Final Production Pipeline
    X_all = df[feature_cols]
    y_all = df["LITHO"].astype(int)

    print("Retraining final model on ALL data...")
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_all, y_all)

    # Predict
    _probs = final_model.predict_proba(X_valid)[:, 1]

    # Apply Optimized Threshold
    valid_predictions = (_probs >= best_thr).astype(int)

    print(f"Done. Prediction mean: {valid_predictions.mean():.3f}")
    return (valid_predictions,)


@app.cell
def _(Path, pd, valid_predictions):
    submission = pd.Series(valid_predictions, name="prediction")
    output_path = Path(__file__).resolve().parent / "pred.csv"
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    submission.head()
    return


@app.cell
def _():
    import marimo as mo
    return


if __name__ == "__main__":
    app.run()
