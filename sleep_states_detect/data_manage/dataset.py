import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from sleep_states_detect.data_manage.load_data import load_data
from sleep_states_detect.data_manage.prepare_data import prepare_data


class SleepDataset(Dataset):
    def __init__(self, input, target, flag):
        self.input = torch.FloatTensor(input)
        self.target = target
        self.flag = torch.FloatTensor(flag)

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        if self.target is not None:
            return (
                self.input[idx],
                torch.FloatTensor(self.target.iloc[idx].values),
                self.flag[idx],
            )
        else:
            return (
                self.input[idx],
                torch.Tensor(),
                self.flag[idx],
            )


class SleepDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path_X="data/X.npy",
        data_path_Y="data/Y.csv",
        batch_size=8,
        num_workers=8,
        train_val_split=0.8,
    ):
        super().__init__()
        self.data_path_X = data_path_X
        self.data_path_Y = data_path_Y
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.num_workers = num_workers

    def prepare_data(self):
        """Скачивание и предварительная обработка данных"""
        load_data(
            link=(
                "https://drive.usercontent.google.com/download?id=151FZM1pJpObdTzfVgDzv"
                "UEF7IXL2g_Hw&export=download&confirm=t&uuid=826277d0-b14c-40aa-ae84-95"
                "06cba5cc66"
            ),
            force=False,
        )
        prepare_data(
            link=(
                "https://drive.google.com/file/d/1a66xTNtj2zSzdATCKGihLX9JlHZuZx8L/view"
                "?usp=sharing"
            ),
            force=False,
        )

    def setup(self, stage: str):
        """
        setup data

        Args:
            stage: 'fit' or 'predict'
        """
        # Загрузка данных
        X = torch.from_numpy(np.load(self.data_path_X)).float()
        df_1min = pd.read_csv(self.data_path_Y)

        # Создание матриц y и mask
        df_y = df_1min.pivot(
            index=["series_id", "date"], columns="time", values="target"
        ).fillna(0)
        df_mask = df_1min.pivot(
            index=["series_id", "date"], columns="time", values="valid_flag"
        ).fillna(0)

        # df_events
        df_events = pd.read_csv("data/" + "train_events.csv").dropna()
        df_events["timestamp"] = pd.to_datetime(
            df_events["timestamp"], utc=True
        ).dt.tz_localize(None)
        df_events["time"] = df_events["timestamp"].dt.time.astype(str)
        df_events["minute_mod15"] = df_events["timestamp"].dt.minute % 15

        if stage == "fit":
            # Получение уникальных серий
            unique_series_ids = df_1min["series_id"].unique()
            np.random.shuffle(unique_series_ids)

            # Деление на train / val series
            split_idx = int(self.train_val_split * len(unique_series_ids))
            train_ids = set(unique_series_ids[:split_idx])
            val_ids = set(unique_series_ids[split_idx:])

            # Определим индексы, соответствующие series_id в train/val
            df_index = df_1min.drop_duplicates(
                subset=["series_id", "date"]
            ).reset_index()

            series_ids = df_index["series_id"].values
            train_indices = [i for i, sid in enumerate(series_ids) if sid in train_ids]
            val_indices = [i for i, sid in enumerate(series_ids) if sid in val_ids]

            # Подмножества датасета
            self.train_dataset = SleepDataset(
                X[train_indices],
                df_y.iloc[train_indices],
                df_mask.iloc[train_indices].to_numpy(),
            )
            self.val_dataset = SleepDataset(
                X[val_indices],
                df_y.iloc[val_indices],
                df_mask.iloc[val_indices].to_numpy(),
            )

            self.train_df_1min = df_1min[df_1min["series_id"].isin(train_ids)]
            self.val_df_1min = df_1min[df_1min["series_id"].isin(val_ids)]
            self.train_df_events = df_events[df_events["series_id"].isin(train_ids)]
            self.val_df_events = df_events[df_events["series_id"].isin(val_ids)]

        if stage == "predict":
            self.predict_dataset = SleepDataset(X, df_y, df_mask.to_numpy())
            self.df_events = df_events
            self.df_1min = df_1min

    def make_results(self, output, dataset_part: str = None):
        """
        make dataframe in kaggle format from time series

        Args:
            output: list with model output (time series)
        Returns:
            pd.df: with columns series_id, step, event, score
        """

        try:
            data_pivot = self.predict_dataset.target
            data_base = self.df_1min
        except AttributeError:
            data_pivot = self.val_dataset.target
            data_base = self.val_df_1min
        if dataset_part == "train":
            data_pivot = self.train_dataset.target
            data_base = self.train_df_1min

        df_pred = pd.DataFrame(
            output,
            index=data_pivot.index[: output.shape[0]],
            columns=data_pivot.columns,
        )
        df_pred = df_pred.stack().reset_index(name="score")
        df_pred = pd.merge(
            data_base[["series_id", "date", "time", "step", "event"]],
            df_pred,
            on=["series_id", "date", "time"],
            how="inner",
        )
        return df_pred

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
