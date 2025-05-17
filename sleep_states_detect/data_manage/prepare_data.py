import gc
import pickle

import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm

from sleep_states_detect.data_manage.load_data import load_data
from sleep_states_detect.utils.utils import check_files_exist


def prepare_data(link: str = None, force: bool = False):
    """Обработка начального датасета и сохранение в удобном виде

    Args:
        link: ссылка на загрузку подготовленных данных (или None, если надо готовить)
        force: проверять ли на существование финальных файлов

    TODO: ест много оперативки, почему-то игнорит swap из-за чего крашится
    """

    if not force and check_files_exist(
        "data/",
        [
            "X.npy",
            "Y.csv",
            "dict_valid_ratio.pkl",
        ],
    ):
        print("данные существуют")
        return

    if link is not None:
        # загружаем готовые данные
        load_data(link=link, force=True)
        return

    gc.collect()

    INPUT_DIR = "data/"

    df_series = pl.read_parquet(INPUT_DIR + "train_series.parquet")
    df_events = pl.read_csv(INPUT_DIR + "train_events.csv")
    df_events = df_events.with_columns(
        pl.col("event").replace({"wakeup": 1.0, "onset": -1.0}).cast(pl.Float32)
    )

    n_unique = df_series.get_column("series_id").n_unique()

    dict_valid_ratio = dict()
    list_feature_array = []
    list_df_1min = []
    for series_id, df in tqdm(
        df_series.group_by("series_id", maintain_order=True), total=n_unique
    ):
        series_id = series_id[0]
        df = (
            df.join(
                df_events.filter(pl.col("series_id") == series_id).select(
                    "timestamp", "event"
                ),
                on="timestamp",
                how="left",
            )
            .with_columns(
                pl.col("timestamp").str.to_datetime(),
                pl.col("event").fill_null(0.0),
            )
            .with_columns(
                pl.col("timestamp").dt.date().cast(str).alias("date"),
                pl.col("timestamp").dt.time().cast(str).alias("time"),
            )
        ).to_pandas()

        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        dup_count = df.groupby(["anglez", "enmo", "time"])["step"].transform("count")
        df["valid_flag"] = (dup_count == 1).astype("float32")
        dict_valid_ratio[series_id] = df["valid_flag"].mean()

        list_feature_array_tmp = []
        df["log_anglez_std"] = np.log(
            df["anglez"].rolling(25, min_periods=1, center=True).std() + 1
        ).astype("float32")
        df["log_enmo"] = np.log(df["enmo"] + 0.01).astype("float32")
        for feature in ["log_anglez_std", "log_enmo", "valid_flag"]:
            df_pivot = df.pivot(
                index=["series_id", "date"], columns="time", values=feature
            )
            feature_array = df_pivot.fillna(0).values
            feature_array_1day_bedore = df_pivot.shift(1).fillna(0).values
            feature_array_1day_after = df_pivot.shift(-1).fillna(0).values
            feature_array = np.concatenate(
                [
                    feature_array_1day_bedore[:, -180 * 12 :],
                    feature_array,
                    feature_array_1day_after[:, : 180 * 12],
                ],
                axis=1,
            )
            list_feature_array_tmp.append(feature_array)
        list_feature_array.append(np.stack(list_feature_array_tmp, axis=1))

        dict_agg = {
            "series_id": "first",
            "date": "first",
            "time": "first",
            "step": "mean",
            "event": "sum",
            "valid_flag": "max",
        }
        df_1min = df.resample("1min", on="timestamp").agg(dict_agg).reset_index()
        df_1min["step"] = df_1min["step"].astype("int32")
        values_event = df_1min["event"].values
        values_target = values_event.copy()
        for j in range(30):
            weight = np.exp(-j / 2.8)
            values_target[: -(j + 1)] += (
                weight * values_event[(j + 1) :]
            )  # shift backward
            if j > 0:
                values_target[j:] += weight * values_event[:-j]  # shift forward
        df_1min["target"] = values_target
        list_df_1min.append(df_1min)

    X = np.concatenate(list_feature_array)
    del list_feature_array
    X = (X - X.min(axis=(0, 2), keepdims=True)) / (
        X.max(axis=(0, 2), keepdims=True) - X.min(axis=(0, 2), keepdims=True)
    )

    df_1min = pd.concat(list_df_1min)

    df_1min.to_csv(INPUT_DIR + "Y.csv")
    np.save(INPUT_DIR + "X.npy", X)
    with open("dict_valid_ratio.pkl", "wb") as f:
        pickle.dump(dict_valid_ratio, f)
