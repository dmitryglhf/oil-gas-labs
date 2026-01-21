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
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from optuna.samplers import TPESampler
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import GroupKFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Resolve root directory
    ROOT_DIR = Path(__file__).resolve().parent.parent

    return (
        CatBoostClassifier,
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
        xgb,
    )


@app.cell
def _(ROOT_DIR, lasio, pd, tqdm):
    # Load training data from LAS files
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

    print(f"Loaded {len(train_data)} samples from {train_data['Well'].nunique()} wells")
    return data_path, train_data


@app.cell
def _(train_data):
    # Basic sanity checks
    train_data["LITHO"] = train_data["LITHO"].astype(int)

    print("Data info:")
    train_data.info()
    print("\nClass distribution:")
    print(train_data["LITHO"].value_counts())
    print(f"\nPositive ratio: {train_data['LITHO'].mean():.3f}")

    print("\nMissing values:")
    print(train_data.isnull().sum())

    train_data.head()
    return


@app.cell
def _(np, pd):
    # Feature engineering as a reusable function (train/val identical)
    def make_features(df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()

        # Ratios and interactions
        df["GR_DENS_ratio"] = df["GR"] / (df["DENS"] + 1e-6)
        df["SP_GR_ratio"] = df["SP"] / (df["GR"] + 1e-6)
        df["DT_DENS_ratio"] = df["DT"] / (df["DENS"] + 1e-6)
        df["GR_DT_product"] = df["GR"] * df["DT"]

        # Normalized per well
        for col in ["SP", "GR", "DT", "DENS"]:
            df[f"{col}_norm"] = df.groupby("Well")[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-6)
            )

        # Rolling stats per well (window=3)
        for col in ["SP", "GR", "DT", "DENS"]:
            df[f"{col}_roll_mean"] = df.groupby("Well")[col].transform(
                lambda x: x.rolling(window=3, min_periods=1, center=True).mean()
            )
            df[f"{col}_roll_std"] = df.groupby("Well")[col].transform(
                lambda x: x.rolling(window=3, min_periods=1, center=True).std()
            )

        df = df.fillna(0)

        # Depth-based
        df["DEPT_norm"] = df.groupby("Well")["DEPT"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
        )

        # Log transforms
        df["GR_log"] = np.log1p(df["GR"].clip(lower=0))
        df["DT_log"] = np.log1p(df["DT"].clip(lower=0))

        return df

    return (make_features,)


@app.cell
def _(make_features, train_data):
    df = make_features(train_data)
    print(f"Features after engineering: {df.shape[1]}")
    df.head()
    return (df,)


@app.cell
def _(df, rd):
    # Split data into train and test sets (70/30 by wells)
    train_part_size = 0.7
    rd.seed(17)

    all_wells = df.Well.unique().tolist()
    train_wells = rd.sample(all_wells, round(len(all_wells) * train_part_size))

    train_set = df.loc[df.Well.isin(train_wells)].reset_index(drop=True)
    test_set = df.loc[~df.Well.isin(train_wells)].reset_index(drop=True)

    print(
        f"Train wells: {len(train_wells)}, Test wells: {len(all_wells) - len(train_wells)}"
    )
    print(f"Train samples: {len(train_set)}, Test samples: {len(test_set)}")
    return test_set, train_set


@app.cell
def _():
    base_features = ["SP", "GR", "DT", "DENS"]
    engineered_features = [
        "GR_DENS_ratio",
        "SP_GR_ratio",
        "DT_DENS_ratio",
        "GR_DT_product",
        "SP_norm",
        "GR_norm",
        "DT_norm",
        "DENS_norm",
        "SP_roll_mean",
        "GR_roll_mean",
        "DT_roll_mean",
        "DENS_roll_mean",
        "SP_roll_std",
        "GR_roll_std",
        "DT_roll_std",
        "DENS_roll_std",
        "DEPT_norm",
        "GR_log",
        "DT_log",
    ]
    feature_cols = base_features + engineered_features
    return (feature_cols,)


@app.cell
def _(feature_cols, test_set, train_set):
    X_train = train_set[feature_cols]
    y_train = train_set["LITHO"].astype(int)
    g_train = train_set["Well"]

    X_test = test_set[feature_cols]
    y_test = test_set["LITHO"].astype(int)

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_test, X_train, g_train, y_test, y_train


@app.cell
def _(
    CatBoostClassifier,
    GroupKFold,
    TPESampler,
    f1_score,
    lgb,
    np,
    optuna,
    xgb,
):
    # --- Model builders ---
    def build_model(name: str, params: dict, seed: int = 42):
        if name == "lgb":
            p = dict(
                objective="binary",
                metric="binary_logloss",
                verbosity=-1,
                random_state=seed,
                n_jobs=-1,
            )
            p.update(params)
            return lgb.LGBMClassifier(**p)

        if name == "xgb":
            p = dict(
                objective="binary:logistic",
                eval_metric="logloss",
                verbosity=0,
                random_state=seed,
                n_jobs=-1,
            )
            p.update(params)
            return xgb.XGBClassifier(**p)

        if name == "cat":
            p = dict(
                loss_function="Logloss",
                verbose=False,
                random_seed=seed,
                allow_writing_files=False,
            )
            p.update(params)
            return CatBoostClassifier(**p)

        raise ValueError(f"Unknown model name: {name}")

    # --- Optuna objectives for each base model (F1 over GroupKFold) ---
    def lgb_objective(trial, X, y, groups, cv):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 150, 600),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", None]
            ),
        }

        scores = []
        for tr_idx, va_idx in cv.split(X, y, groups):
            model = build_model("lgb", params)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = model.predict(X.iloc[va_idx])
            scores.append(f1_score(y.iloc[va_idx], pred))
        return float(np.mean(scores))

    def xgb_objective(trial, X, y, groups, cv):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 150, 700),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 6.0),
        }

        scores = []
        for tr_idx, va_idx in cv.split(X, y, groups):
            model = build_model("xgb", params)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = model.predict(X.iloc[va_idx])
            scores.append(f1_score(y.iloc[va_idx], pred))
        return float(np.mean(scores))

    def cat_objective(trial, X, y, groups, cv):
        params = {
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "iterations": trial.suggest_int("iterations", 200, 800),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 20.0, log=True),
            "auto_class_weights": trial.suggest_categorical(
                "auto_class_weights", ["Balanced", "SqrtBalanced", None]
            ),
        }

        scores = []
        for tr_idx, va_idx in cv.split(X, y, groups):
            model = build_model("cat", params)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = model.predict(X.iloc[va_idx])
            scores.append(f1_score(y.iloc[va_idx], pred))
        return float(np.mean(scores))

    # --- OOF probabilities helper ---
    def oof_proba(
        model_name: str,
        best_params: dict,
        X,
        y,
        groups,
        n_splits: int = 3,
        seed: int = 42,
    ):
        cv = GroupKFold(n_splits=n_splits)
        oof = np.zeros(len(X), dtype=float)

        for tr_idx, va_idx in cv.split(X, y, groups):
            model = build_model(model_name, best_params, seed=seed)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            proba = model.predict_proba(X.iloc[va_idx])[:, 1]
            oof[va_idx] = proba

        return oof

    # --- Tune base models + tune blending weights (and threshold) ---
    def tune_and_blend(
        X, y, groups, n_trials_base: int = 30, n_trials_blend: int = 80, seed: int = 42
    ):
        cv = GroupKFold(n_splits=3)

        # 1) Tune each base model
        print("=" * 70)
        print("Tuning base models (GroupKFold by Well)")
        print("=" * 70)

        sampler = TPESampler(seed=seed)

        # LightGBM
        print("\n[1/3] Tuning LightGBM...")
        study_lgb = optuna.create_study(direction="maximize", sampler=sampler)
        study_lgb.optimize(
            lambda t: lgb_objective(t, X, y, groups, cv),
            n_trials=n_trials_base,
            show_progress_bar=True,
        )
        best_lgb = study_lgb.best_params
        print(f"Best LGB CV F1: {study_lgb.best_value:.4f}")

        # XGBoost
        print("\n[2/3] Tuning XGBoost...")
        study_xgb = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=seed)
        )
        study_xgb.optimize(
            lambda t: xgb_objective(t, X, y, groups, cv),
            n_trials=n_trials_base,
            show_progress_bar=True,
        )
        best_xgb = study_xgb.best_params
        print(f"Best XGB CV F1: {study_xgb.best_value:.4f}")

        # CatBoost
        print("\n[3/3] Tuning CatBoost...")
        study_cat = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=seed)
        )
        study_cat.optimize(
            lambda t: cat_objective(t, X, y, groups, cv),
            n_trials=n_trials_base,
            show_progress_bar=True,
        )
        best_cat = study_cat.best_params
        print(f"Best CAT CV F1: {study_cat.best_value:.4f}")

        # 2) Compute OOF probabilities for blending
        print("\n" + "=" * 70)
        print("Computing OOF probabilities for blending...")
        print("=" * 70)

        oof_lgb = oof_proba("lgb", best_lgb, X, y, groups, n_splits=3, seed=seed)
        oof_xgb = oof_proba("xgb", best_xgb, X, y, groups, n_splits=3, seed=seed)
        oof_cat = oof_proba("cat", best_cat, X, y, groups, n_splits=3, seed=seed)

        # 3) Tune blending weights + threshold on OOF
        def blend_objective(trial):
            w1 = trial.suggest_float("w_lgb", 0.0, 1.0)
            w2 = trial.suggest_float("w_xgb", 0.0, 1.0)
            w3 = trial.suggest_float("w_cat", 0.0, 1.0)
            s = w1 + w2 + w3 + 1e-12
            w1, w2, w3 = w1 / s, w2 / s, w3 / s

            thr = trial.suggest_float("threshold", 0.2, 0.8)

            p = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
            pred = (p >= thr).astype(int)
            return f1_score(y, pred)

        print("\n" + "=" * 70)
        print("Optimizing blending weights (Optuna on OOF predictions)...")
        print("=" * 70)

        study_blend = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=seed)
        )
        study_blend.optimize(
            blend_objective, n_trials=n_trials_blend, show_progress_bar=True
        )

        best_blend = study_blend.best_params
        # normalize final weights
        s = best_blend["w_lgb"] + best_blend["w_xgb"] + best_blend["w_cat"] + 1e-12
        best_blend["w_lgb"] /= s
        best_blend["w_xgb"] /= s
        best_blend["w_cat"] /= s

        print("\nBest blend OOF F1:", f"{study_blend.best_value:.4f}")
        print("Best blend params:", best_blend)

        return {
            "best_params": {"lgb": best_lgb, "xgb": best_xgb, "cat": best_cat},
            "blend": best_blend,
            "oof": {"lgb": oof_lgb, "xgb": oof_xgb, "cat": oof_cat},
        }

    return build_model, tune_and_blend


@app.cell
def _(X_train, g_train, tune_and_blend, y_train):
    # Run tuning + blending optimization on train split
    ensemble_artifacts = tune_and_blend(
        X_train, y_train, g_train, n_trials_base=30, n_trials_blend=80, seed=42
    )
    return (ensemble_artifacts,)


@app.cell
def _(
    X_test,
    X_train,
    accuracy_score,
    build_model,
    classification_report,
    ensemble_artifacts,
    f1_score,
    np,
    y_test,
    y_train,
):
    # локальные имена (с префиксом _) можно переиспользовать в других ячейках
    _best_params = ensemble_artifacts["best_params"]
    _blend = ensemble_artifacts["blend"]

    model_lgb_test = build_model("lgb", _best_params["lgb"], seed=42)
    model_xgb_test = build_model("xgb", _best_params["xgb"], seed=42)
    model_cat_test = build_model("cat", _best_params["cat"], seed=42)

    model_lgb_test.fit(X_train, y_train)
    model_xgb_test.fit(X_train, y_train)
    model_cat_test.fit(X_train, y_train)

    _p_lgb = model_lgb_test.predict_proba(X_test)[:, 1]
    _p_xgb = model_xgb_test.predict_proba(X_test)[:, 1]
    _p_cat = model_cat_test.predict_proba(X_test)[:, 1]

    _p = _blend["w_lgb"] * _p_lgb + _blend["w_xgb"] * _p_xgb + _blend["w_cat"] * _p_cat
    pred_test = (_p >= _blend["threshold"]).astype(int)

    f1_test = f1_score(y_test, pred_test)
    acc_test = accuracy_score(y_test, pred_test)

    print("Ensemble on TEST wells")
    print(f"F1: {f1_test:.4f}, Accuracy: {acc_test:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, pred_test))

    return acc_test, f1_test, model_cat_test, model_lgb_test, model_xgb_test, pred_test


@app.cell
def _(data_path, make_features, pd):
    # Load validation data and apply same feature engineering
    validation_data = pd.read_csv(data_path / "Shestakovo_validation.csv")
    print(f"Validation data shape: {validation_data.shape}")
    print(f"Validation wells: {validation_data['Well'].nunique()}")

    val_df = make_features(validation_data)
    print(f"Validation features shape: {val_df.shape}")
    val_df.head()
    return val_df, validation_data


@app.cell
def _(
    build_model,
    df,
    ensemble_artifacts,
    feature_cols,
    np,
    val_df,
):
    X_all = df[feature_cols]
    y_all = df["LITHO"].astype(int)

    _best_params = ensemble_artifacts["best_params"]
    _blend = ensemble_artifacts["blend"]

    final_lgb = build_model("lgb", _best_params["lgb"], seed=42)
    final_xgb = build_model("xgb", _best_params["xgb"], seed=42)
    final_cat = build_model("cat", _best_params["cat"], seed=42)

    print("Training final base models on ALL labeled data...")
    final_lgb.fit(X_all, y_all)
    final_xgb.fit(X_all, y_all)
    final_cat.fit(X_all, y_all)

    X_valid = val_df[feature_cols]

    _p_lgb = final_lgb.predict_proba(X_valid)[:, 1]
    _p_xgb = final_xgb.predict_proba(X_valid)[:, 1]
    _p_cat = final_cat.predict_proba(X_valid)[:, 1]

    _p = _blend["w_lgb"] * _p_lgb + _blend["w_xgb"] * _p_xgb + _blend["w_cat"] * _p_cat
    valid_predictions = (_p >= _blend["threshold"]).astype(int)

    print(f"Predictions made: {len(valid_predictions)}")
    return final_cat, final_lgb, final_xgb, valid_predictions


@app.cell
def _():
    # Set user name for submission (optional)
    user_name = "Solution"
    return (user_name,)


@app.cell
def _(Path, pd, valid_predictions):
    # Save predictions
    submission = pd.Series(valid_predictions, name="prediction")
    output_path = Path(__file__).resolve().parent / "pred.csv"
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    submission.head(20)
    return output_path, submission


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
