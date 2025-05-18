import hydra
from omegaconf import DictConfig

from sleep_states_detect.data_manage.download_data import download_data
from sleep_states_detect.utils.utils import check_files_exist


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def download_kaggle_data(cfg: DictConfig):
    if cfg["download_data"]["force_load"] or not check_files_exist(
        cfg["file_names"]["data_folder"],
        [
            cfg["file_names"]["kaggle_train_events"],
            cfg["file_names"]["kaggle_train_series"],
        ],
    ):
        download_data(
            link=cfg["download_data"]["link"],
            data_dir=cfg["file_names"]["data_folder"],
        )
    else:
        print("kaggle data is already loaded")
