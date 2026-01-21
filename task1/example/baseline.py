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
    from sklearn.metrics import mean_absolute_error
    from sklearn.linear_model import RANSACRegressor as Ridge
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    import missingno as msno
    from pathlib import Path

    # Increase the number of displayed columns and rows
    pd.set_option("display.max_columns", 300)
    pd.set_option("display.max_rows", 300)
    np.set_printoptions(suppress=True)
    return (
        MinMaxScaler,
        Path,
        Ridge,
        bn,
        mean_absolute_error,
        msno,
        np,
        pd,
        plt,
        reduce,
    )


@app.cell
def _(Path, pd):
    # Load the train data
    ROOT_DIR = Path(__file__).parent.parent / "example"
    DATA_DIR = ROOT_DIR /"data"
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    train
    return DATA_DIR, train


@app.cell
def _(train):
    # Pivot the train data to get it in the required format
    data = train.pivot_table(index="date", columns="cat", values="oil")
    data.head()
    return (data,)


@app.cell
def _(data, msno):
    # Visualize the missing data using Missingno
    msno.matrix(data, sparkline=False, figsize=(15, 15))
    return


@app.cell
def _(data):
    # Get the maximum date in the dataset
    max_date_in_dataset = data.index.max()
    max_date_in_dataset
    return (max_date_in_dataset,)


@app.cell
def _(max_date_in_dataset, pd):
    # Create subsequent date values according to the specified horizon
    HORIZON = 24
    future_dates = pd.date_range(start=max_date_in_dataset, periods=HORIZON + 1, freq="MS")
    future_dates
    return (future_dates,)


@app.cell
def _(data, future_dates):
    # Add the new date values to the data index
    new_index = data.index.append(future_dates)
    new_index
    return (new_index,)


@app.cell
def _(data, new_index):
    data_1 = data.reindex(new_index)
    data_1
    return (data_1,)


@app.cell
def _(data_1):
    # Initialize rolling statistics dataframes with original data
    rolling_mean_24 = data_1.copy()
    rolling_std_24 = data_1.copy()
    rolling_median_24 = data_1.copy()
    rolling_diff_24 = data_1.copy()
    rolling_exp_weighted_mean = data_1.copy()
    return (
        rolling_diff_24,
        rolling_exp_weighted_mean,
        rolling_mean_24,
        rolling_median_24,
        rolling_std_24,
    )


@app.cell
def _(bn, np, pd):
    # Function to compute rolling statistics
    def bottleneck_stats(
        series, periods=1, min_count=1, window=4, fillna=None, aggfunc="mean"
    ):
        """
        Function to compute rolling statistics using the bottleneck library
        This function calculates various rolling statistics such as mean, standard deviation, sum, median, difference, and rank.
        The function can be customized to compute any of these statistics with a given window size, minimum count of non-NaN values, and fill NaN values with a specified method.
        This is a very flexible function that can be used to generate a wide range of features from time series data.

        Параметры
        ----------
        series:
            pandas.Series
        periods: int, по умолчанию 1
            Порядок лага, с которым вычисляем скользящие
            статистики.
        min_periods: int, по умолчанию 1
            Минимальное количество наблюдений в окне для
            вычисления скользящих статистик.
        window: int, по умолчанию 4
            Ширина окна. Не должна быть меньше
            горизонта прогнозирования.
        fast: bool, по умолчанию True
            Режим вычислений скользящих статистик.
        fillna, int, по умолчанию 0
            Стратегия импутации пропусков.
        aggfunc, string, по умолчанию 'mean'
            Агрегирующая функция.
        """

        def shift(xs, n):
            return np.concatenate((np.full(n, np.nan), xs[:-n]))

        arr = series.values
        arr = shift(xs=arr, n=periods)
        if aggfunc == "mean":
            arr = bn.move_mean(arr, window=window, min_count=min_count)
        if aggfunc == "std":
            arr = bn.move_std(arr, window=window, min_count=min_count)
        if aggfunc == "sum":
            arr = bn.move_sum(arr, window=window, min_count=min_count)
        if aggfunc == "median":
            arr = bn.move_median(arr, window=window, min_count=min_count)
        if aggfunc == "diff":
            min_arr = bn.move_min(arr, window=window, min_count=min_count)
            max_arr = bn.move_max(arr, window=window, min_count=min_count)
            diff_arr = max_arr - min_arr
            diff_arr = np.insert(diff_arr, 0, diff_arr[-1])
            arr = np.delete(diff_arr, -1)
        if aggfunc == "rank":
            arr = bn.move_rank(arr, window=window, min_count=min_count)

        features = pd.Series(arr)
        features.index = series.index

        # импутируем пропуски
        if fillna is not None:
            features.fillna(fillna, inplace=True)

        return features
    return (bottleneck_stats,)


@app.cell
def _(
    bottleneck_stats,
    data_1,
    rolling_diff_24,
    rolling_exp_weighted_mean,
    rolling_mean_24,
    rolling_median_24,
    rolling_std_24,
):
    # Calculate rolling statistics for each segment in the data
    # This loop applies the bottleneck_stats function to each segment in the data to generate rolling statistics.
    segments_list = data_1.columns.tolist()
    for i in segments_list:
        rolling_mean_24[i] = bottleneck_stats(rolling_mean_24[i], window=24, min_count=1)
        rolling_std_24[i] = bottleneck_stats(rolling_std_24[i], window=24, min_count=1, aggfunc='std')
        rolling_median_24[i] = bottleneck_stats(rolling_median_24[i], window=24, min_count=1, aggfunc='median')
        rolling_diff_24[i] = bottleneck_stats(rolling_diff_24[i], window=24, min_count=1, aggfunc='diff')
        rolling_exp_weighted_mean[i] = rolling_exp_weighted_mean[i].ewm(alpha=0.15).mean()
    return


@app.cell
def _(
    rolling_diff_24,
    rolling_exp_weighted_mean,
    rolling_mean_24,
    rolling_median_24,
    rolling_std_24,
):
    # Fill NaN values in rolling statistics dataframes
    # The NaN values in the rolling statistics dataframes are filled with zeros.
    rolling_mean_24.fillna(0, inplace=True)
    rolling_std_24.fillna(0, inplace=True)
    rolling_median_24.fillna(0, inplace=True)
    rolling_diff_24.fillna(0, inplace=True)
    rolling_exp_weighted_mean.fillna(0, inplace=True)
    return


@app.cell
def _(rolling_mean_24):
    # Assigning the date index to a new column in each of the rolling statistics dataframes
    # This is done to prepare the dataframes for melting (converting from wide format to long format) in the next steps.
    rolling_mean_24["date"] = rolling_mean_24.index
    rolling_mean_24
    return


@app.cell
def _(
    rolling_diff_24,
    rolling_exp_weighted_mean,
    rolling_median_24,
    rolling_std_24,
):
    rolling_std_24["date"] = rolling_std_24.index
    rolling_median_24["date"] = rolling_median_24.index
    rolling_diff_24["date"] = rolling_diff_24.index
    rolling_exp_weighted_mean["date"] = rolling_exp_weighted_mean.index
    return


@app.cell
def _(rolling_mean_24):
    # Melt the rolling statistics dataframes from wide format to long format
    # Melting is the process of reshaping data where each row is a unique id-date pair and each statistic has its own column. This is a common format for time series data.
    rolling_mean_24_1 = rolling_mean_24.melt(id_vars='date', var_name='cat', value_name='rolling_mean_24')
    rolling_mean_24_1
    return (rolling_mean_24_1,)


@app.cell
def _(
    rolling_diff_24,
    rolling_exp_weighted_mean,
    rolling_median_24,
    rolling_std_24,
):
    rolling_std_24_1 = rolling_std_24.melt(id_vars='date', var_name='cat', value_name='rolling_std_24')
    rolling_median_24_1 = rolling_median_24.melt(id_vars='date', var_name='cat', value_name='rolling_median_24')
    rolling_diff_24_1 = rolling_diff_24.melt(id_vars='date', var_name='cat', value_name='rolling_diff_24')
    rolling_exp_weighted_mean_1 = rolling_exp_weighted_mean.melt(id_vars='date', var_name='cat', value_name='rolling_exp_weighted_mean')
    return (
        rolling_diff_24_1,
        rolling_exp_weighted_mean_1,
        rolling_median_24_1,
        rolling_std_24_1,
    )


@app.cell
def _(DATA_DIR, pd):
    # Load the test data
    # The test data is loaded from a csv file into a pandas dataframe.
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
    test
    return (test,)


@app.cell
def _(pd, test, train):
    # Concatenate the train and test data into a single dataframe
    # The train and test data are concatenated along the row axis (axis=0). This means that the test data is appended at the end of the train data.
    data_2 = pd.concat([train, test], axis=0)
    data_2.reset_index(drop=True, inplace=True)
    data_2
    return (data_2,)


@app.cell
def _(
    data_2,
    pd,
    reduce,
    rolling_diff_24_1,
    rolling_exp_weighted_mean_1,
    rolling_mean_24_1,
    rolling_median_24_1,
    rolling_std_24_1,
):
    # Merge all dataframes into a single dataframe
    # The dataframes containing the original data and the rolling statistics are merged into a single dataframe. The merging is done on the 'cat' and 'date' columns, which are common to all dataframes.
    data_frames = [data_2, rolling_mean_24_1, rolling_std_24_1, rolling_median_24_1, rolling_diff_24_1, rolling_exp_weighted_mean_1]
    data_3 = reduce(lambda left, right: pd.merge(left, right, how='left', on=['cat', 'date']), data_frames)
    data_3
    return (data_3,)


@app.cell
def _(data_3, plt):
    _cats = ['hw-3', 'well-175', 'well-180', 'inj-1', 'inj-30', 'inj-12']
    for _cat in _cats:
        plt.title(_cat)
        _df = data_3[data_3['cat'] == _cat].copy()
        _label = 'oil'
        plt.plot(_df['date'], _df[_label], label=_label)
        _label = 'rolling_mean_24'
        plt.plot(_df['date'], _df[_label], label=_label)
        _label = 'rolling_diff_24'
        plt.plot(_df['date'], _df[_label], label=_label)
        _label = 'water_inj'
        plt.plot(_df['date'], _df[_label], label=_label)
        plt.legend()
        plt.grid(ls='--')
        plt.show()
    return


@app.cell
def _(data_3):
    # Split the data into train and test sets based on the date
    # The data is split into train and test sets such that all data before February 2048 is used for training and all data from February 2048 is used for testing.
    train_1 = data_3[data_3['date'] < '2048-02-01']
    test_1 = data_3[data_3['date'] >= '2048-02-01']
    return test_1, train_1


@app.cell
def _(train_1):
    # Further split the train data into training and validation sets
    # The train data is further split into training and validation sets. The training set includes all data before February 2046 and the validation set includes all data from February 2046.
    tr = train_1[train_1['date'] < '2046-02-01']
    val = train_1[train_1['date'] >= '2046-02-01']
    return tr, val


@app.cell
def _(tr):
    tr.head()
    return


@app.cell
def _(tr, val):
    # Drop irrelevant columns from training and validation sets
    # Some columns that are not useful for modeling are dropped from the training and validation sets.
    tr_1 = tr[tr['group'] != 'I']
    val_1 = val[val['group'] != 'I']
    tr_ = tr_1.copy()
    val_ = val_1.copy()
    cols_to_drop = ['cat', 'date', 'group', 'water_inj', 'gor', 'gas', 'watercut', 'water', 'bhp']
    tr_1 = tr_1.drop(cols_to_drop, axis=1)
    val_1 = val_1.drop(cols_to_drop, axis=1)
    y_tr = tr_1.pop('oil')
    y_val = val_1.pop('oil')
    return cols_to_drop, tr_, tr_1, val_, val_1, y_tr, y_val


@app.cell
def _(tr_1):
    # Get variable with names of numerical columns
    num_cols = tr_1.select_dtypes(exclude='object').columns
    num_cols
    return (num_cols,)


@app.cell
def _(MinMaxScaler, num_cols, tr_1, train_1, val_1):
    _scaler = MinMaxScaler()
    _scaler.fit(train_1[num_cols])
    tr_1[num_cols] = _scaler.transform(tr_1[num_cols])
    val_1[num_cols] = _scaler.transform(val_1[num_cols])
    return


@app.cell
def _(Ridge, mean_absolute_error, tr_1, val_1, y_tr, y_val):
    # Fit a Ridge regression model and make predictions
    # A Ridge regression model is fit on the training data and used to make predictions on the training and validation data. The mean absolute error (MAE) of the predictions is calculated and printed.
    model = Ridge()
    model.fit(tr_1, y_tr)
    tr_fcst = model.predict(tr_1)
    val_fcst = model.predict(val_1)
    mae_tr = mean_absolute_error(y_tr, tr_fcst)
    mae_val = mean_absolute_error(y_val, val_fcst)
    print('MAE train:', f'{mae_tr:.2f}')
    print('MAE validation:', f'{mae_val:.2f}')
    return model, tr_fcst, val_fcst


@app.cell
def _(tr_, tr_fcst, val_, val_fcst):
    # Assign the predictions to new columns in the original training and validation dataframes
    # The predictions made by the Ridge regression model are assigned to new columns named 'fcst' in the original training and validation dataframes.
    tr_["fcst"] = tr_fcst
    val_["fcst"] = val_fcst
    return


@app.cell
def _(plt, tr_, val_):
    # Visualize the true and predicted values for some segments
    # This loop generates time series plots for some selected segments. Each plot includes the true and predicted values for the training and validation data.
    _cats = ['hw-3', 'well-175', 'well-180']
    for _cat in _cats:
        plt.title(_cat)
        _df = tr_[tr_['cat'] == _cat].copy()
        _label = 'oil'
        plt.plot(_df['date'], _df[_label], label='train_true')
        _label = 'fcst'
        plt.plot(_df['date'], _df[_label], label='train_fcst')
        _df = val_[val_['cat'] == _cat].copy()
        _label = 'oil'
        plt.plot(_df['date'], _df[_label], label='val_fcst_true')
        _label = 'fcst'
        plt.plot(_df['date'], _df[_label], label='val_fcst')
        plt.legend()
        plt.grid(ls='--')
        plt.show()
    return


@app.cell
def _(test_1, train_1):
    # Preprocess the train and test data for final model training
    # The 'group' column is dropped from the train and test data. This is because the 'group' column is not useful for the final model training.
    train_2 = train_1[train_1['group'] != 'I']
    test_2 = test_1[test_1['group'] != 'I']
    return test_2, train_2


@app.cell
def _(test_2):
    # Extract the 'cat' and 'date' columns from the test data and store in a new dataframe
    # The 'cat' and 'date' columns are needed for the final submission file.
    test_ = test_2[['cat', 'date']].copy()
    return (test_,)


@app.cell
def _(train_2):
    # Pop the target variable from the train data
    # The 'oil' column is the target variable and is popped from the train data.
    y_train = train_2.pop('oil')
    return (y_train,)


@app.cell
def _(cols_to_drop, np, test_2, train_2):
    # Drop irrelevant columns from train and test data
    # Some columns that are not useful for modeling are dropped from the train and test data.
    train_3 = train_2.drop(cols_to_drop, axis=1)
    test_3 = test_2.drop(np.append(cols_to_drop, ['oil']), axis=1)
    return test_3, train_3


@app.cell
def _(MinMaxScaler, num_cols, test_3, train_3):
    _scaler = MinMaxScaler()
    _scaler.fit(train_3[num_cols])
    train_3[num_cols] = _scaler.transform(train_3[num_cols])
    test_3[num_cols] = _scaler.transform(test_3[num_cols])
    return


@app.cell
def _(model, test_3, train_3, y_train):
    # Fit the Ridge model on the train data and make predictions on the test data
    model.fit(train_3, y_train)
    fcst = model.predict(test_3)
    return (fcst,)


@app.cell
def _(DATA_DIR, fcst, test_):
    # Assign the predictions to a new column in the test data
    # The predictions made by the Ridge regression model are assigned to a new column named 'fcst' in the test data.
    test_["fcst"] = fcst

    # Save the test data with predictions to a csv file
    # The test data along with the predictions is saved to a csv file for submission.
    test_.to_csv(DATA_DIR / "fcst.csv", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Сделать прогноз с MAE<96
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
