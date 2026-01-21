import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    # Import necessary libraries
    import pandas as pd
    import numpy as np
    import bottleneck as bn
    from functools import reduce
    from pathlib import Path
    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    import lightgbm as lgb

    pd.set_option("display.max_columns", 300)
    pd.set_option("display.max_rows", 100)
    np.set_printoptions(suppress=True)

    # Resolve root directory
    ROOT_DIR = Path(__file__).resolve().parent.parent
    return Path, ROOT_DIR, bn, np, pd, reduce


@app.cell
def _(ROOT_DIR, pd):
    # Load the data
    data_path = ROOT_DIR / "task1" / "example" / "data"
    train = pd.read_csv(data_path / "train.csv", parse_dates=["date"])
    test = pd.read_csv(data_path / "test.csv", parse_dates=["date"])
    coords = pd.read_csv(data_path / "coords.csv")
    return coords, test, train


@app.cell
def _(train):
    # Explore the data
    print(f"Train shape: {train.shape}")
    print(f"Date range: {train['date'].min()} to {train['date'].max()}")
    print(f"Unique categories: {train['cat'].nunique()}")
    print(f"Groups: {train['group'].unique()}")
    train.head()
    return


@app.cell
def _(train):
    # Pivot the train data to get it in wide format for rolling calculations
    data = train.pivot_table(index="date", columns="cat", values="oil")
    data.head()
    return (data,)


@app.cell
def _(data, pd):
    # Create future dates for the forecast horizon
    HORIZON = 24
    max_date = data.index.max()
    future_dates = pd.date_range(start=max_date, periods=HORIZON + 1, freq="MS")
    new_index = data.index.append(future_dates)
    data_extended = data.reindex(new_index)
    print(f"Extended data shape: {data_extended.shape}")
    return (data_extended,)


@app.cell
def _(bn, np, pd):
    # Function to compute rolling statistics using bottleneck
    def bottleneck_stats(
        series, periods=1, min_count=1, window=4, fillna=None, aggfunc="mean"
    ):
        def shift(xs, n):
            return np.concatenate((np.full(n, np.nan), xs[:-n]))

        arr = series.values
        arr = shift(xs=arr, n=periods)

        if aggfunc == "mean":
            arr = bn.move_mean(arr, window=window, min_count=min_count)
        elif aggfunc == "std":
            arr = bn.move_std(arr, window=window, min_count=min_count)
        elif aggfunc == "sum":
            arr = bn.move_sum(arr, window=window, min_count=min_count)
        elif aggfunc == "median":
            arr = bn.move_median(arr, window=window, min_count=min_count)
        elif aggfunc == "diff":
            min_arr = bn.move_min(arr, window=window, min_count=min_count)
            max_arr = bn.move_max(arr, window=window, min_count=min_count)
            arr = max_arr - min_arr
        elif aggfunc == "min":
            arr = bn.move_min(arr, window=window, min_count=min_count)
        elif aggfunc == "max":
            arr = bn.move_max(arr, window=window, min_count=min_count)

        features = pd.Series(arr)
        features.index = series.index

        if fillna is not None:
            features.fillna(fillna, inplace=True)

        return features
    return (bottleneck_stats,)


@app.cell
def _(bottleneck_stats, data_extended):
    # Calculate rolling statistics with multiple windows
    segments_list = data_extended.columns.tolist()

    # Window 24 features
    rolling_mean_24 = data_extended.copy()
    rolling_std_24 = data_extended.copy()
    rolling_median_24 = data_extended.copy()
    rolling_diff_24 = data_extended.copy()
    rolling_min_24 = data_extended.copy()
    rolling_max_24 = data_extended.copy()

    # Window 12 features
    rolling_mean_12 = data_extended.copy()
    rolling_std_12 = data_extended.copy()
    rolling_diff_12 = data_extended.copy()

    # Window 6 features
    rolling_mean_6 = data_extended.copy()
    rolling_std_6 = data_extended.copy()

    # Exponential weighted mean
    rolling_ewm = data_extended.copy()

    for i in segments_list:
        # Window 24
        rolling_mean_24[i] = bottleneck_stats(rolling_mean_24[i], window=24, min_count=1, aggfunc='mean')
        rolling_std_24[i] = bottleneck_stats(rolling_std_24[i], window=24, min_count=1, aggfunc='std')
        rolling_median_24[i] = bottleneck_stats(rolling_median_24[i], window=24, min_count=1, aggfunc='median')
        rolling_diff_24[i] = bottleneck_stats(rolling_diff_24[i], window=24, min_count=1, aggfunc='diff')
        rolling_min_24[i] = bottleneck_stats(rolling_min_24[i], window=24, min_count=1, aggfunc='min')
        rolling_max_24[i] = bottleneck_stats(rolling_max_24[i], window=24, min_count=1, aggfunc='max')

        # Window 12
        rolling_mean_12[i] = bottleneck_stats(rolling_mean_12[i], window=12, min_count=1, aggfunc='mean')
        rolling_std_12[i] = bottleneck_stats(rolling_std_12[i], window=12, min_count=1, aggfunc='std')
        rolling_diff_12[i] = bottleneck_stats(rolling_diff_12[i], window=12, min_count=1, aggfunc='diff')

        # Window 6
        rolling_mean_6[i] = bottleneck_stats(rolling_mean_6[i], window=6, min_count=1, aggfunc='mean')
        rolling_std_6[i] = bottleneck_stats(rolling_std_6[i], window=6, min_count=1, aggfunc='std')

        # EWM
        rolling_ewm[i] = rolling_ewm[i].ewm(alpha=0.15).mean()

    print("Rolling statistics calculated")
    return (
        rolling_diff_12,
        rolling_diff_24,
        rolling_ewm,
        rolling_max_24,
        rolling_mean_12,
        rolling_mean_24,
        rolling_mean_6,
        rolling_median_24,
        rolling_min_24,
        rolling_std_12,
        rolling_std_24,
        rolling_std_6,
    )


@app.cell
def _(
    rolling_diff_12,
    rolling_diff_24,
    rolling_ewm,
    rolling_max_24,
    rolling_mean_12,
    rolling_mean_24,
    rolling_mean_6,
    rolling_median_24,
    rolling_min_24,
    rolling_std_12,
    rolling_std_24,
    rolling_std_6,
):
    # Fill NaN values with 0
    rolling_dfs = [
        rolling_mean_24, rolling_std_24, rolling_median_24, rolling_diff_24,
        rolling_min_24, rolling_max_24, rolling_mean_12, rolling_std_12,
        rolling_diff_12, rolling_mean_6, rolling_std_6, rolling_ewm
    ]
    for df in rolling_dfs:
        df.fillna(0, inplace=True)
    return


@app.cell
def _(
    rolling_diff_12,
    rolling_diff_24,
    rolling_ewm,
    rolling_max_24,
    rolling_mean_12,
    rolling_mean_24,
    rolling_mean_6,
    rolling_median_24,
    rolling_min_24,
    rolling_std_12,
    rolling_std_24,
    rolling_std_6,
):
    # Add date column for melting
    rolling_mean_24["date"] = rolling_mean_24.index
    rolling_std_24["date"] = rolling_std_24.index
    rolling_median_24["date"] = rolling_median_24.index
    rolling_diff_24["date"] = rolling_diff_24.index
    rolling_min_24["date"] = rolling_min_24.index
    rolling_max_24["date"] = rolling_max_24.index
    rolling_mean_12["date"] = rolling_mean_12.index
    rolling_std_12["date"] = rolling_std_12.index
    rolling_diff_12["date"] = rolling_diff_12.index
    rolling_mean_6["date"] = rolling_mean_6.index
    rolling_std_6["date"] = rolling_std_6.index
    rolling_ewm["date"] = rolling_ewm.index
    return


@app.cell
def _(
    rolling_diff_12,
    rolling_diff_24,
    rolling_ewm,
    rolling_max_24,
    rolling_mean_12,
    rolling_mean_24,
    rolling_mean_6,
    rolling_median_24,
    rolling_min_24,
    rolling_std_12,
    rolling_std_24,
    rolling_std_6,
):
    # Melt all rolling statistics dataframes
    rolling_mean_24_m = rolling_mean_24.melt(id_vars='date', var_name='cat', value_name='rolling_mean_24')
    rolling_std_24_m = rolling_std_24.melt(id_vars='date', var_name='cat', value_name='rolling_std_24')
    rolling_median_24_m = rolling_median_24.melt(id_vars='date', var_name='cat', value_name='rolling_median_24')
    rolling_diff_24_m = rolling_diff_24.melt(id_vars='date', var_name='cat', value_name='rolling_diff_24')
    rolling_min_24_m = rolling_min_24.melt(id_vars='date', var_name='cat', value_name='rolling_min_24')
    rolling_max_24_m = rolling_max_24.melt(id_vars='date', var_name='cat', value_name='rolling_max_24')
    rolling_mean_12_m = rolling_mean_12.melt(id_vars='date', var_name='cat', value_name='rolling_mean_12')
    rolling_std_12_m = rolling_std_12.melt(id_vars='date', var_name='cat', value_name='rolling_std_12')
    rolling_diff_12_m = rolling_diff_12.melt(id_vars='date', var_name='cat', value_name='rolling_diff_12')
    rolling_mean_6_m = rolling_mean_6.melt(id_vars='date', var_name='cat', value_name='rolling_mean_6')
    rolling_std_6_m = rolling_std_6.melt(id_vars='date', var_name='cat', value_name='rolling_std_6')
    rolling_ewm_m = rolling_ewm.melt(id_vars='date', var_name='cat', value_name='rolling_ewm')
    return (
        rolling_diff_12_m,
        rolling_diff_24_m,
        rolling_ewm_m,
        rolling_max_24_m,
        rolling_mean_12_m,
        rolling_mean_24_m,
        rolling_mean_6_m,
        rolling_median_24_m,
        rolling_min_24_m,
        rolling_std_12_m,
        rolling_std_24_m,
        rolling_std_6_m,
    )


@app.cell
def _(pd, test, train):
    # Concatenate train and test data
    data_combined = pd.concat([train, test], axis=0)
    data_combined.reset_index(drop=True, inplace=True)
    print(f"Combined data shape: {data_combined.shape}")
    return (data_combined,)


@app.cell
def _(
    coords,
    data_combined,
    pd,
    reduce,
    rolling_diff_12_m,
    rolling_diff_24_m,
    rolling_ewm_m,
    rolling_max_24_m,
    rolling_mean_12_m,
    rolling_mean_24_m,
    rolling_mean_6_m,
    rolling_median_24_m,
    rolling_min_24_m,
    rolling_std_12_m,
    rolling_std_24_m,
    rolling_std_6_m,
):
    # Merge all features into a single dataframe
    data_frames = [
        data_combined,
        rolling_mean_24_m, rolling_std_24_m, rolling_median_24_m, rolling_diff_24_m,
        rolling_min_24_m, rolling_max_24_m,
        rolling_mean_12_m, rolling_std_12_m, rolling_diff_12_m,
        rolling_mean_6_m, rolling_std_6_m,
        rolling_ewm_m
    ]
    data_merged = reduce(lambda left, right: pd.merge(left, right, how='left', on=['cat', 'date']), data_frames)

    # Add coordinates
    data_merged = pd.merge(data_merged, coords, how='left', on='cat')

    print(f"Merged data shape: {data_merged.shape}")
    data_merged.head()
    return (data_merged,)


@app.cell
def _(data_merged):
    # Add time-based features
    data_merged['month'] = data_merged['date'].dt.month
    data_merged['year'] = data_merged['date'].dt.year
    data_merged['quarter'] = data_merged['date'].dt.quarter

    # Create trend feature (months since start)
    min_date = data_merged['date'].min()
    data_merged['trend'] = ((data_merged['date'] - min_date).dt.days / 30).astype(int)

    print("Time features added")
    data_merged.head()
    return


@app.cell
def _(data_merged):
    # Split data into train and test sets
    train_data = data_merged[data_merged['date'] < '2048-02-01'].copy()
    test_data = data_merged[data_merged['date'] >= '2048-02-01'].copy()

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    return test_data, train_data


@app.cell
def _():
    # Define columns to drop for modeling
    cols_to_drop = ['cat', 'date', 'group', 'water_inj', 'gor', 'gas', 'watercut', 'water', 'bhp']
    return (cols_to_drop,)


@app.cell
def _(cols_to_drop, tr_filtered, val_filtered):
    # Prepare training and validation data
    tr_copy = tr_filtered.copy()
    val_copy = val_filtered.copy()

    X_tr = tr_copy.drop(cols_to_drop + ['oil'], axis=1)
    y_tr = tr_copy['oil']

    X_val = val_copy.drop(cols_to_drop + ['oil'], axis=1)
    y_val = val_copy['oil']

    print(f"X_tr shape: {X_tr.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"Features: {X_tr.columns.tolist()}")
    return (tr_copy,)


@app.cell
def _():
    from autogluon.tabular import TabularPredictor
    return (TabularPredictor,)


@app.cell
def _(TabularPredictor, tr_copy):
    predictor = TabularPredictor(label="oil", eval_metric="mean_absolute_error").fit(tr_copy)
    return (predictor,)


@app.cell
def _():
    return


@app.cell
def _(cols_to_drop, test_data, train_data):
    # Prepare final training on full train data and predict test
    train_final = train_data[train_data['group'] != 'I'].copy()
    test_final = test_data[test_data['group'] != 'I'].copy()

    X_train_final = train_final.drop(cols_to_drop + ['oil'], axis=1)
    y_train_final = train_final['oil']

    X_test_final = test_final.drop(cols_to_drop + ['oil'], axis=1)

    print(f"Final training shape: {X_train_final.shape}")
    print(f"Final test shape: {X_test_final.shape}")
    return X_test_final, test_final


@app.cell
def _(X_test_final, predictor):
    y_pred = predictor.predict(X_test_final)
    return


@app.cell
def _(Path, test_final, test_predictions):
    # Create submission file
    submission = test_final[['cat', 'date']].copy()
    submission['fcst'] = test_predictions

    # Ensure non-negative predictions
    submission['fcst'] = submission['fcst'].clip(lower=0)

    output_path = Path(__file__).resolve().parent / "pred.csv"
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    submission.head(20)
    return (submission,)


@app.cell
def _(submission):
    # Summary statistics of predictions
    print("Prediction statistics:")
    print(submission['fcst'].describe())
    return


@app.cell
def _():
    import marimo as mo
    return


if __name__ == "__main__":
    app.run()
