import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    # Import libraries
    import pandas as pd
    import numpy as np
    import random as rd
    import seaborn as sns
    from pathlib import Path
    from tqdm import tqdm
    import lasio
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    import lightgbm as lgb
    import warnings
    warnings.filterwarnings('ignore')

    # Resolve root directory
    ROOT_DIR = Path(__file__).resolve().parent.parent
    return (
        Path,
        ROOT_DIR,
        RandomForestClassifier,
        accuracy_score,
        classification_report,
        f1_score,
        lasio,
        lgb,
        np,
        pd,
        rd,
        tqdm,
    )


@app.cell
def _(ROOT_DIR, lasio, pd, tqdm):
    # Load training data from LAS files
    train_data = pd.DataFrame()
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
    # Explore data
    print("Data info:")
    train_data.info()
    print("\nClass distribution:")
    print(train_data["LITHO"].value_counts())
    print(f"\nClass ratio: {train_data['LITHO'].mean():.3f}")
    train_data.head()
    return


@app.cell
def _(train_data):
    # Convert LITHO to int and check for missing values
    train_data["LITHO"] = train_data["LITHO"].astype(int)
    print("Missing values:")
    print(train_data.isnull().sum())
    train_data.describe()
    return


@app.cell
def _(np, train_data):
    # Feature engineering: add derived features
    df = train_data.copy()

    # Ratios and interactions
    df['GR_DENS_ratio'] = df['GR'] / (df['DENS'] + 1e-6)
    df['SP_GR_ratio'] = df['SP'] / (df['GR'] + 1e-6)
    df['DT_DENS_ratio'] = df['DT'] / (df['DENS'] + 1e-6)
    df['GR_DT_product'] = df['GR'] * df['DT']

    # Normalized features within each well
    for col in ['SP', 'GR', 'DT', 'DENS']:
        df[f'{col}_norm'] = df.groupby('Well')[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6)
        )

    # Rolling statistics per well (window=3)
    for col in ['SP', 'GR', 'DT', 'DENS']:
        df[f'{col}_roll_mean'] = df.groupby('Well')[col].transform(
            lambda x: x.rolling(window=3, min_periods=1, center=True).mean()
        )
        df[f'{col}_roll_std'] = df.groupby('Well')[col].transform(
            lambda x: x.rolling(window=3, min_periods=1, center=True).std()
        )

    # Fill NaN values from rolling calculations
    df = df.fillna(0)

    # Depth-based features
    df['DEPT_norm'] = df.groupby('Well')['DEPT'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
    )

    # Log transforms for skewed features
    df['GR_log'] = np.log1p(df['GR'].clip(lower=0))
    df['DT_log'] = np.log1p(df['DT'].clip(lower=0))

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

    train_set = df.loc[df.Well.isin(train_wells)]
    test_set = df.loc[~df.Well.isin(train_wells)]

    print(f"Train wells: {len(train_wells)}, Test wells: {len(all_wells) - len(train_wells)}")
    print(f"Train samples: {len(train_set)}, Test samples: {len(test_set)}")
    return test_set, train_set


@app.cell
def _():
    # Define feature columns
    base_features = ['SP', 'GR', 'DT', 'DENS']
    engineered_features = [
        'GR_DENS_ratio', 'SP_GR_ratio', 'DT_DENS_ratio', 'GR_DT_product',
        'SP_norm', 'GR_norm', 'DT_norm', 'DENS_norm',
        'SP_roll_mean', 'GR_roll_mean', 'DT_roll_mean', 'DENS_roll_mean',
        'SP_roll_std', 'GR_roll_std', 'DT_roll_std', 'DENS_roll_std',
        'DEPT_norm', 'GR_log', 'DT_log'
    ]
    feature_cols = base_features + engineered_features
    return (feature_cols,)


@app.cell
def _(feature_cols, test_set, train_set):
    # Prepare train and test data
    X_train = train_set[feature_cols]
    y_train = train_set["LITHO"]

    X_test = test_set[feature_cols]
    y_test = test_set["LITHO"]

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    return X_test, X_train, y_test, y_train


@app.cell
def _():
    # Additional imports for hyperparameter tuning
    import optuna
    from optuna.samplers import TPESampler
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from sklearn.model_selection import GroupKFold
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return CatBoostClassifier, GroupKFold, TPESampler, optuna, xgb


@app.cell
def _(CatBoostClassifier, lgb, np, xgb):
    # Objective functions for Optuna (Classification)
    def lgb_clf_objective(trial, X, y, groups, cv):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        }
        model = lgb.LGBMClassifier(**params)
        # Manual GroupKFold cross-validation with F1 scoring
        scores = []
        for train_idx, val_idx in cv.split(X, y, groups):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_train_cv, y_train_cv)
            y_pred = model.predict(X_val_cv)
            from sklearn.metrics import f1_score
            scores.append(f1_score(y_val_cv, y_pred))
        return np.mean(scores)

    def xgb_clf_objective(trial, X, y, groups, cv):
        params = {
            'objective': 'binary:logistic',
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 5),
            'verbosity': 0,
        }
        model = xgb.XGBClassifier(**params)
        scores = []
        for train_idx, val_idx in cv.split(X, y, groups):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_train_cv, y_train_cv)
            y_pred = model.predict(X_val_cv)
            from sklearn.metrics import f1_score
            scores.append(f1_score(y_val_cv, y_pred))
        return np.mean(scores)

    def catboost_clf_objective(trial, X, y, groups, cv):
        params = {
            'loss_function': 'Logloss',
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'iterations': trial.suggest_int('iterations', 100, 300),
            'auto_class_weights': trial.suggest_categorical('auto_class_weights', ['Balanced', 'SqrtBalanced']),
            'verbose': False,
        }
        model = CatBoostClassifier(**params)
        scores = []
        for train_idx, val_idx in cv.split(X, y, groups):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_train_cv, y_train_cv)
            y_pred = model.predict(X_val_cv)
            from sklearn.metrics import f1_score
            scores.append(f1_score(y_val_cv, y_pred))
        return np.mean(scores)

    return catboost_clf_objective, lgb_clf_objective, xgb_clf_objective


@app.cell
def _(
    CatBoostClassifier,
    GroupKFold,
    TPESampler,
    catboost_clf_objective,
    lgb,
    lgb_clf_objective,
    optuna,
    pd,
    xgb,
    xgb_clf_objective,
):
    # Function to tune and compare classification models
    def tune_and_compare_clf(X, y, groups, n_trials=30):
        cv = GroupKFold(n_splits=3)
        results = {}

        print("=" * 60)
        print("Starting hyperparameter tuning with Optuna (30 trials per model)")
        print("Using GroupKFold with grouping by Well")
        print("=" * 60)

        # LightGBM
        print("\n[1/3] Tuning LightGBM...")
        study_lgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study_lgb.optimize(lambda t: lgb_clf_objective(t, X, y, groups, cv), n_trials=n_trials, show_progress_bar=True)
        results['LightGBM'] = {
            'f1': study_lgb.best_value,
            'params': study_lgb.best_params,
            'model_class': lgb.LGBMClassifier
        }
        print(f"  Best F1: {study_lgb.best_value:.4f}")

        # XGBoost
        print("\n[2/3] Tuning XGBoost...")
        study_xgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study_xgb.optimize(lambda t: xgb_clf_objective(t, X, y, groups, cv), n_trials=n_trials, show_progress_bar=True)
        results['XGBoost'] = {
            'f1': study_xgb.best_value,
            'params': study_xgb.best_params,
            'model_class': xgb.XGBClassifier
        }
        print(f"  Best F1: {study_xgb.best_value:.4f}")

        # CatBoost
        print("\n[3/3] Tuning CatBoost...")
        study_catboost = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study_catboost.optimize(lambda t: catboost_clf_objective(t, X, y, groups, cv), n_trials=n_trials, show_progress_bar=True)
        results['CatBoost'] = {
            'f1': study_catboost.best_value,
            'params': study_catboost.best_params,
            'model_class': CatBoostClassifier
        }
        print(f"  Best F1: {study_catboost.best_value:.4f}")

        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'F1 Score': [results[m]['f1'] for m in results],
            'Best Params': [str(results[m]['params']) for m in results]
        }).sort_values('F1 Score', ascending=False)

        print("\n" + "=" * 60)
        print("Model Comparison Results (sorted by F1 Score)")
        print("=" * 60)
        print(comparison_df.to_string(index=False))

        return results

    return (tune_and_compare_clf,)


@app.cell
def _(X_train, train_set, tune_and_compare_clf, y_train):
    # Run hyperparameter tuning and model comparison
    groups_train = train_set['Well']
    comparison_results = tune_and_compare_clf(X_train, y_train, groups_train, n_trials=30)
    return comparison_results, groups_train


@app.cell
def _(CatBoostClassifier, X_test, X_train, accuracy_score, comparison_results, f1_score, lgb, xgb, y_test, y_train):
    # Train the best model with optimal hyperparameters
    best_model_name = max(comparison_results, key=lambda k: comparison_results[k]['f1'])
    best_params = comparison_results[best_model_name]['params']

    print(f"\nBest model: {best_model_name}")
    print(f"Best parameters: {best_params}")

    # Create and train the best model
    if best_model_name == 'LightGBM':
        best_model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            verbosity=-1,
            **best_params
        )
    elif best_model_name == 'XGBoost':
        best_model = xgb.XGBClassifier(
            objective='binary:logistic',
            verbosity=0,
            **best_params
        )
    else:  # CatBoost
        best_model = CatBoostClassifier(
            loss_function='Logloss',
            verbose=False,
            **best_params
        )

    best_model.fit(X_train, y_train)

    # Make predictions
    best_pred = best_model.predict(X_test)
    best_f1 = f1_score(y_test, best_pred)
    best_acc = accuracy_score(y_test, best_pred)

    print(f'\n{best_model_name} (tuned) - F1: {best_f1:.4f}, Accuracy: {best_acc:.4f}')
    return best_acc, best_f1, best_model, best_model_name, best_params, best_pred


@app.cell
def _(X_train, best_model, best_model_name, pd):
    # Feature importance
    import matplotlib.pyplot as plt

    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance['feature'][:15], feature_importance['importance'][:15])
    plt.xlabel('Importance')
    plt.title(f'Top 15 Feature Importances ({best_model_name})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    feature_importance
    return


@app.cell
def _(best_model_name, best_pred, classification_report, y_test):
    # Detailed classification report
    print(f"{best_model_name} Classification Report:")
    print(classification_report(y_test, best_pred))
    return


@app.cell
def _(CatBoostClassifier, best_model_name, best_params, df, feature_cols, lgb, xgb):
    # Retrain on all data for final prediction with best hyperparameters
    X_all = df[feature_cols]
    y_all = df["LITHO"]

    print(f"\nTraining final {best_model_name} model on all data...")

    if best_model_name == 'LightGBM':
        final_model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            verbosity=-1,
            **best_params
        )
    elif best_model_name == 'XGBoost':
        final_model = xgb.XGBClassifier(
            objective='binary:logistic',
            verbosity=0,
            **best_params
        )
    else:  # CatBoost
        final_model = CatBoostClassifier(
            loss_function='Logloss',
            verbose=False,
            **best_params
        )

    final_model.fit(X_all, y_all)
    print("Final model trained on all data")
    return (final_model,)


@app.cell
def _(data_path, np, pd):
    # Load validation data
    validation_data = pd.read_csv(data_path / "Shestakovo_validation.csv")
    print(f"Validation data shape: {validation_data.shape}")
    print(f"Validation wells: {validation_data['Well'].nunique()}")

    # Feature engineering for validation data (same as training)
    val_df = validation_data.copy()

    # Ratios and interactions
    val_df['GR_DENS_ratio'] = val_df['GR'] / (val_df['DENS'] + 1e-6)
    val_df['SP_GR_ratio'] = val_df['SP'] / (val_df['GR'] + 1e-6)
    val_df['DT_DENS_ratio'] = val_df['DT'] / (val_df['DENS'] + 1e-6)
    val_df['GR_DT_product'] = val_df['GR'] * val_df['DT']

    # Normalized features within each well
    for c in ['SP', 'GR', 'DT', 'DENS']:
        val_df[f'{c}_norm'] = val_df.groupby('Well')[c].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6)
        )

    # Rolling statistics per well
    for c in ['SP', 'GR', 'DT', 'DENS']:
        val_df[f'{c}_roll_mean'] = val_df.groupby('Well')[c].transform(
            lambda x: x.rolling(window=3, min_periods=1, center=True).mean()
        )
        val_df[f'{c}_roll_std'] = val_df.groupby('Well')[c].transform(
            lambda x: x.rolling(window=3, min_periods=1, center=True).std()
        )

    val_df = val_df.fillna(0)

    # Depth-based features
    val_df['DEPT_norm'] = val_df.groupby('Well')['DEPT'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
    )

    # Log transforms
    val_df['GR_log'] = np.log1p(val_df['GR'].clip(lower=0))
    val_df['DT_log'] = np.log1p(val_df['DT'].clip(lower=0))

    print(f"Validation features shape: {val_df.shape}")
    val_df.head()
    return (val_df,)


@app.cell
def _(feature_cols, final_model, pd, val_df):
    # Make predictions on validation data
    X_valid = val_df[feature_cols]
    valid_predictions = final_model.predict(X_valid)

    print(f"Predictions made: {len(valid_predictions)}")
    print(f"Predicted class distribution:")
    print(pd.Series(valid_predictions).value_counts())
    return (valid_predictions,)


@app.cell
def _():
    # Set user name for submission
    user_name = "Solution"
    return


@app.cell
def _(Path, pd, valid_predictions):
    # Save predictions
    submission = pd.Series(valid_predictions, name="prediction")
    output_path = Path(__file__).resolve().parent / "pred.csv"
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    submission.head(20)
    return


@app.cell
def _():
    import marimo as mo
    return


if __name__ == "__main__":
    app.run()
