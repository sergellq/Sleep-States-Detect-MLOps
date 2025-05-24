import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from sleep_states_detect.data_manage.dataset import SleepDataModule
from sleep_states_detect.data_manage.dvc_load import dvc_load
from sleep_states_detect.models.unet1d_lightning import UNet1dLightning


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train_main(cfg: DictConfig):
    dvc_load(cfg["data"])
    OmegaConf.resolve(cfg)

    # Загрузка данных
    data_module = SleepDataModule(cfg)

    # Модель
    model = UNet1dLightning(cfg["model_params"])

    # Логгер для TensorBoard
    logger = TensorBoardLogger("lightning_logs", name="sleep_state_detection")

    # Колбэки

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",  # метрика для отслеживания
        dirpath=cfg["train_params"]["model_save_dir"],  # куда сохранять модель
        filename=cfg["train_params"]["model_save_name"],  # имя файла
        save_top_k=1,  # сохранить только лучший чекпоинт
        mode="min",  # 'min' для loss, 'max' для accuracy
        save_weights_only=True,  # сохраняем только веса, а не весь объект
    )
    early_stop_callback = EarlyStopping(monitor="valid_loss", patience=5, mode="min")

    # Обучение
    trainer = Trainer(
        max_epochs=cfg["model_params"]["max_epochs"],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator="auto",
        devices="auto",
    )
    trainer.fit(model, datamodule=data_module)
