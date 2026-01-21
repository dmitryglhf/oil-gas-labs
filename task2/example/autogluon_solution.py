import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    import pandas as pd
    import lasio
    from tqdm import tqdm
    from autogluon.tabular import TabularPredictor
    return Path, TabularPredictor, lasio, pd, tqdm


@app.cell
def _(Path):
    ROOT = Path(__file__).parent
    DATA_PATH = ROOT / "data"
    TRAIN_PATH = DATA_PATH / "train_test"
    return DATA_PATH, ROOT, TRAIN_PATH


@app.cell
def _(TRAIN_PATH, lasio, pd, tqdm):
    las_frames = []
    for las_file in tqdm(list(TRAIN_PATH.glob("*.las"))):
        las_data = lasio.read(las_file).df().reset_index()
        las_data["Well"] = las_file.stem
        las_frames.append(las_data)

    train_raw = pd.concat(las_frames, ignore_index=True)
    train_raw.columns = ["DEPT", "SP", "GR", "DT", "DENS", "LITHO", "Well"]
    train_raw["LITHO"] = train_raw["LITHO"].astype(int)
    return las_frames, train_raw


@app.cell
def _(DATA_PATH, pd):
    validation_raw = pd.read_csv(DATA_PATH / "Shestakovo_validation.csv")
    return (validation_raw,)


@app.cell
def _(train_raw):
    FEATURE_COLS = ["SP", "GR", "DT", "DENS"]
    TARGET_COL = "LITHO"
    train_features = train_raw[FEATURE_COLS + [TARGET_COL]]
    return FEATURE_COLS, TARGET_COL, train_features


@app.cell
def _(FEATURE_COLS, validation_raw):
    validation_features = validation_raw[FEATURE_COLS]
    return (validation_features,)


@app.cell
def _(ROOT, TARGET_COL, TabularPredictor, train_features):
    predictor = TabularPredictor(
        label=TARGET_COL,
        eval_metric="f1",
        path=ROOT / "autogluon_model"
    ).fit(
        train_features,
        presets="high_quality",
    )
    return (predictor,)


@app.cell
def _(predictor, validation_features):
    predictions = predictor.predict(validation_features)
    return (predictions,)


@app.cell
def _(ROOT, pd, predictions):
    USER_NAME = "autogluon"
    submission = pd.Series(predictions, name=f"{USER_NAME}_prediction")
    submission.to_csv(ROOT / f"{USER_NAME}_prediction.csv", index=False)
    submission
    return USER_NAME, submission


if __name__ == "__main__":
    app.run()
