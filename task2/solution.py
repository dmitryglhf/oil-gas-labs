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
        GradientBoostingClassifier,
        Path,
        ROOT_DIR,
        RandomForestClassifier,
        StandardScaler,
        accuracy_score,
        classification_report,
        f1_score,
        lasio,
        lgb,
        np,
        pd,
        rd,
        sns,
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
    return data_path, dfs, train_data, train_test_path


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
    return all_wells, test_set, train_set, train_wells


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
    return base_features, engineered_features, feature_cols


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
def _(X_test, X_train, accuracy_score, f1_score, lgb, y_test, y_train):
    # Train LightGBM model
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 300,
        'class_weight': 'balanced',
    }

    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train, y_train)

    lgb_pred = lgb_model.predict(X_test)
    lgb_f1 = f1_score(y_test, lgb_pred)
    lgb_acc = accuracy_score(y_test, lgb_pred)

    print(f"LightGBM - F1: {lgb_f1:.4f}, Accuracy: {lgb_acc:.4f}")
    return lgb_acc, lgb_f1, lgb_model, lgb_params, lgb_pred


@app.cell
def _(
    RandomForestClassifier,
    X_test,
    X_train,
    accuracy_score,
    f1_score,
    y_test,
    y_train,
):
    # Train Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_f1 = f1_score(y_test, rf_pred)
    rf_acc = accuracy_score(y_test, rf_pred)

    print(f"Random Forest - F1: {rf_f1:.4f}, Accuracy: {rf_acc:.4f}")
    return rf_acc, rf_f1, rf_model, rf_pred


@app.cell
def _(X_train, lgb_model, pd, plt):
    # Feature importance
    import matplotlib.pyplot as plt

    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance['feature'][:15], feature_importance['importance'][:15])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importances (LightGBM)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    feature_importance
    return feature_importance, plt


@app.cell
def _(classification_report, lgb_pred, rf_pred, y_test):
    # Detailed classification report
    print("LightGBM Classification Report:")
    print(classification_report(y_test, lgb_pred))

    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf_pred))
    return


@app.cell
def _(lgb_f1, lgb_model, rf_f1, rf_model):
    # Select best model
    if lgb_f1 >= rf_f1:
        best_model = lgb_model
        best_model_name = "LightGBM"
    else:
        best_model = rf_model
        best_model_name = "Random Forest"

    print(f"Best model: {best_model_name}")
    return best_model, best_model_name


@app.cell
def _(df, feature_cols, lgb, y_train):
    # Retrain on all data for final prediction
    X_all = df[feature_cols]
    y_all = df["LITHO"]

    final_model = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        n_estimators=300,
        class_weight='balanced',
    )
    final_model.fit(X_all, y_all)
    print("Final model trained on all data")
    return X_all, final_model, y_all


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
    for col in ['SP', 'GR', 'DT', 'DENS']:
        val_df[f'{col}_norm'] = val_df.groupby('Well')[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6)
        )

    # Rolling statistics per well
    for col in ['SP', 'GR', 'DT', 'DENS']:
        val_df[f'{col}_roll_mean'] = val_df.groupby('Well')[col].transform(
            lambda x: x.rolling(window=3, min_periods=1, center=True).mean()
        )
        val_df[f'{col}_roll_std'] = val_df.groupby('Well')[col].transform(
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
    return val_df, validation_data


@app.cell
def _(feature_cols, final_model, pd, val_df):
    # Make predictions on validation data
    X_valid = val_df[feature_cols]
    valid_predictions = final_model.predict(X_valid)

    print(f"Predictions made: {len(valid_predictions)}")
    print(f"Predicted class distribution:")
    print(pd.Series(valid_predictions).value_counts())
    return X_valid, valid_predictions


@app.cell
def _():
    # Set user name for submission
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
    return (submission,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
