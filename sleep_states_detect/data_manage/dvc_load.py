from dvc.repo import Repo
from omegaconf import DictConfig

from sleep_states_detect.data_manage.download_data import download_data
from sleep_states_detect.utils.utils import check_files_exist


def dvc_load(cfg: DictConfig):
    try:
        if not check_files_exist(cfg["dvc_folder"], ["dvcstore"]):
            download_data(cfg["dvc_link"], cfg["dvc_folder"])
        repo = Repo(".")
        repo.pull()
    except Exception as e:
        print(f"{e} exception, no dvc, load from google drive")
