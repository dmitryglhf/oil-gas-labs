import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import bottleneck as bn
    from functools import reduce
    from autogluon.tabular import TabularPredictor
    return Path, TabularPredictor, bn, np, pd, reduce


@app.cell
def _(Path, pd):
    ROOT = Path(__file__).parent
    DATA_PATH = ROOT / "data"
    train_raw = pd.read_csv(DATA_PATH / "train.csv", parse_dates=["date"])
    test_raw = pd.read_csv(DATA_PATH / "test.csv", parse_dates=["date"])
    return DATA_PATH, ROOT, test_raw, train_raw


@app.cell
def _(train_raw):
    oil_pivot = train_raw.pivot_table(index="date", columns="cat", values="oil")
    return (oil_pivot,)


@app.cell
def _(oil_pivot, pd):
    HORIZON = 24
    max_date = oil_pivot.index.max()
    future_dates = pd.date_range(start=max_date, periods=HORIZON + 1, freq="MS")
    extended_index = oil_pivot.index.append(future_dates)
    oil_extended = oil_pivot.reindex(extended_index)
    return (oil_extended,)


@app.cell
def _(oil_extended):
    rolling_mean = oil_extended.copy()
    rolling_std = oil_extended.copy()
    rolling_median = oil_extended.copy()
    rolling_diff = oil_extended.copy()
    rolling_ewm = oil_extended.copy()
    return rolling_diff, rolling_ewm, rolling_mean, rolling_median, rolling_std


@app.cell
def _(bn, np, pd):
    def compute_rolling_stats(series, periods=1, min_count=1, window=24, aggfunc="mean"):
        def shift(xs, n):
            return np.concatenate((np.full(n, np.nan), xs[:-n]))

        arr = series.values
        arr = shift(xs=arr, n=periods)

        if aggfunc == "mean":
            arr = bn.move_mean(arr, window=window, min_count=min_count)
        elif aggfunc == "std":
            arr = bn.move_std(arr, window=window, min_count=min_count)
        elif aggfunc == "median":
            arr = bn.move_median(arr, window=window, min_count=min_count)
        elif aggfunc == "diff":
            min_arr = bn.move_min(arr, window=window, min_count=min_count)
            max_arr = bn.move_max(arr, window=window, min_count=min_count)
            diff_arr = max_arr - min_arr
            diff_arr = np.insert(diff_arr, 0, diff_arr[-1])
            arr = np.delete(diff_arr, -1)

        result = pd.Series(arr, index=series.index)
        return result
    return (compute_rolling_stats,)


@app.cell
def _(
    compute_rolling_stats,
    oil_extended,
    rolling_diff,
    rolling_ewm,
    rolling_mean,
    rolling_median,
    rolling_std,
):
    segments = oil_extended.columns.tolist()
    for seg in segments:
        rolling_mean[seg] = compute_rolling_stats(rolling_mean[seg], window=24, aggfunc="mean")
        rolling_std[seg] = compute_rolling_stats(rolling_std[seg], window=24, aggfunc="std")
        rolling_median[seg] = compute_rolling_stats(rolling_median[seg], window=24, aggfunc="median")
        rolling_diff[seg] = compute_rolling_stats(rolling_diff[seg], window=24, aggfunc="diff")
        rolling_ewm[seg] = rolling_ewm[seg].ewm(alpha=0.15).mean()
    return


@app.cell
def _(rolling_diff, rolling_ewm, rolling_mean, rolling_median, rolling_std):
    rolling_mean.fillna(0, inplace=True)
    rolling_std.fillna(0, inplace=True)
    rolling_median.fillna(0, inplace=True)
    rolling_diff.fillna(0, inplace=True)
    rolling_ewm.fillna(0, inplace=True)
    return


@app.cell
def _(rolling_diff, rolling_ewm, rolling_mean, rolling_median, rolling_std):
    rolling_mean["date"] = rolling_mean.index
    rolling_std["date"] = rolling_std.index
    rolling_median["date"] = rolling_median.index
    rolling_diff["date"] = rolling_diff.index
    rolling_ewm["date"] = rolling_ewm.index
    return


@app.cell
def _(rolling_diff, rolling_ewm, rolling_mean, rolling_median, rolling_std):
    mean_melted = rolling_mean.melt(id_vars="date", var_name="cat", value_name="rolling_mean_24")
    std_melted = rolling_std.melt(id_vars="date", var_name="cat", value_name="rolling_std_24")
    median_melted = rolling_median.melt(id_vars="date", var_name="cat", value_name="rolling_median_24")
    diff_melted = rolling_diff.melt(id_vars="date", var_name="cat", value_name="rolling_diff_24")
    ewm_melted = rolling_ewm.melt(id_vars="date", var_name="cat", value_name="rolling_ewm")
    return diff_melted, ewm_melted, mean_melted, median_melted, std_melted


@app.cell
def _(pd, test_raw, train_raw):
    full_data = pd.concat([train_raw, test_raw], axis=0)
    full_data.reset_index(drop=True, inplace=True)
    return (full_data,)


@app.cell
def _(
    diff_melted,
    ewm_melted,
    full_data,
    mean_melted,
    median_melted,
    pd,
    reduce,
    std_melted,
):
    feature_frames = [full_data, mean_melted, std_melted, median_melted, diff_melted, ewm_melted]
    merged_data = reduce(lambda left, right: pd.merge(left, right, how="left", on=["cat", "date"]), feature_frames)
    return (merged_data,)


@app.cell
def _(merged_data):
    train_full = merged_data[merged_data["date"] < "2048-02-01"]
    test_full = merged_data[merged_data["date"] >= "2048-02-01"]
    return test_full, train_full


@app.cell
def _(train_full):
    train_producers = train_full[train_full["group"] != "I"].copy()
    return (train_producers,)


@app.cell
def _(test_full):
    test_producers = test_full[test_full["group"] != "I"].copy()
    submission_template = test_producers[["cat", "date"]].copy()
    return submission_template, test_producers


@app.cell
def _(train_producers):
    COLS_TO_DROP = ["cat", "date", "group", "water_inj", "gor", "gas", "watercut", "water", "bhp"]
    train_features = train_producers.drop(COLS_TO_DROP, axis=1)
    return COLS_TO_DROP, train_features


@app.cell
def _(COLS_TO_DROP, np, test_producers):
    test_features = test_producers.drop(np.append(COLS_TO_DROP, ["oil"]), axis=1)
    return (test_features,)


@app.cell
def _(ROOT, TabularPredictor, train_features):
    predictor = TabularPredictor(
        label="oil",
        eval_metric="mean_absolute_error",
        path=ROOT / "autogluon_model"
    ).fit(
        train_features,
        presets="high_quality",  # high_quality
        # time_limit=300
    )
    return (predictor,)


@app.cell
def _(predictor, test_features):
    predictions = predictor.predict(test_features)
    return (predictions,)


@app.cell
def _(DATA_PATH, predictions, submission_template):
    submission_template["fcst"] = predictions.values
    submission_template.to_csv(DATA_PATH / "fcst.csv", index=False)
    submission_template
    return


if __name__ == "__main__":
    app.run()
