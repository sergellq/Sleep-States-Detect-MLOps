import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from sleep_states_detect.data_manage.dataset import SleepDataModule
from sleep_states_detect.data_manage.dvc_load import dvc_load
from sleep_states_detect.metrics.find_peaks import predict_peaks
from sleep_states_detect.models.unet1d_lightning import UNet1dLightning
from sleep_states_detect.utils.utils import get_latest_checkpoint


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def infer_main(cfg: DictConfig):
    dvc_load(cfg["data_load"])
    OmegaConf.resolve(cfg)

    # Загрузка данных
    data_module = SleepDataModule(
        cfg["data_load"], cfg["data_infer"], cfg["dataset_params"]
    )
    data_module.setup("infer")

    # Загрузка модели
    model = UNet1dLightning.load_from_checkpoint(
        get_latest_checkpoint(cfg["train_params"]["model_save_dir"])
    )

    # Предсказание
    trainer = Trainer()
    predictions = trainer.predict(model, datamodule=data_module)

    # Постпроцессинг данных
    data = data_module.make_results(torch.cat(predictions, dim=0))
    print(data.shape)
    data = predict_peaks(data)
    print(data.shape)

    # Сохранение результата
    data.to_csv(cfg["results"])
