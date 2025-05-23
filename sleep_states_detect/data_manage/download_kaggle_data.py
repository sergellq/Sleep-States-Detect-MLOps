# import hydra
from omegaconf import DictConfig

from sleep_states_detect.data_manage.download_data import download_data
from sleep_states_detect.utils.utils import check_files_exist


def download_kaggle_data(cfg: DictConfig):
    if cfg["base_force_load"] or not check_files_exist(
        cfg["data_folder"],
        [
            cfg["kaggle_train_events"],
            cfg["kaggle_train_series"],
        ],
    ):
        download_data(
            link=cfg["base_link"],
            data_dir=cfg["data_folder"],
        )
    else:
        print("kaggle data is already loaded")
