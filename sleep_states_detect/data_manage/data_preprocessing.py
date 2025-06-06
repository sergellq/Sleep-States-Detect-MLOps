import numpy as np
import pandas as pd
import polars as pl
from omegaconf import DictConfig
from tqdm.auto import tqdm

from sleep_states_detect.utils.utils import check_files_exist


def data_preprocessing(cfg_load: DictConfig, cfg_names: DictConfig):
    """Обработка начального датасета и сохранение в удобном виде

    Args:
        link: ссылка на загрузку подготовленных данных (или None, если надо готовить)
        force: проверять ли на существование финальных файлов

    TODO: ест много оперативки (~20 gb), почему-то игнорит swap из-за чего крашится
          (лотерея 50/50)
    """
    if check_files_exist(
        cfg_load["data_folder"],
        [
            cfg_names["mask_data"],
            cfg_names["train_data"],
        ],
    ):
        print("preprocessed data is already exists")
        return

    df_series = pl.read_parquet(
        cfg_load["data_folder"] + cfg_names["kaggle_train_series"],
        low_memory=True,
    )
    if cfg_names["mode"] == "train":
        df_events = pl.read_csv(
            cfg_load["data_folder"] + cfg_names["kaggle_train_events"]
        )
        df_events = df_events.with_columns(
            pl.col("event").replace({"wakeup": 1.0, "onset": -1.0}).cast(pl.Float32)
        )
    elif cfg_names["mode"] == "infer":
        df_events = pd.DataFrame(
            columns=["series_id", "night", "event", "step", "timestamp"]
        )
        for series_id, _ in df_series.group_by("series_id", maintain_order=True):
            df_events.loc[len(df_events)] = [
                series_id[0],
                1,  # night
                "onset",  # event
                None,  # step
                None,  # timestamp
            ]
            df_events.loc[len(df_events)] = [
                series_id[0],
                1,  # night
                "wakeup",  # event
                None,  # step
                None,  # timestamp
            ]
        df_events.to_csv(cfg_load["data_folder"] + cfg_names["kaggle_train_events"])
        df_events = pl.from_pandas(df_events)

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
        df["event"] = df["event"].astype(float)

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

    # df_events
    df_events = pd.read_csv(
        cfg_load["data_folder"] + cfg_names["kaggle_train_events"]
    ).dropna()
    df_events["timestamp"] = pd.to_datetime(
        df_events["timestamp"], utc=True
    ).dt.tz_localize(None)
    df_events["time"] = df_events["timestamp"].dt.time.astype(str)
    df_events["minute_mod15"] = df_events["timestamp"].dt.minute % 15

    # Создание матриц y и mask
    df_y = df_1min.pivot(
        index=["series_id", "date"], columns="time", values="target"
    ).fillna(0)
    df_mask = df_1min.pivot(
        index=["series_id", "date"], columns="time", values="valid_flag"
    ).fillna(0)

    df_events.to_csv(cfg_load["data_folder"] + cfg_names["postprocessed_target"])
    df_y.to_csv(cfg_load["data_folder"] + cfg_names["target_data"])
    df_mask.to_csv(cfg_load["data_folder"] + cfg_names["mask_data"])
    df_1min.to_csv(cfg_load["data_folder"] + cfg_names["base_data"])

    np.save(cfg_load["data_folder"] + cfg_names["train_data"], X)
