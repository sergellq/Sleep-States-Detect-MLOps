from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from sleep_states_detect.data_manage.dataset import SleepDataModule
from sleep_states_detect.models.unet1d_lightning import UNet1dLightning


def train_main():
    # Загрузка данных
    data_module = SleepDataModule()

    # Модель
    model = UNet1dLightning(
        input_channels=3,
        initial_channels=72,
        initial_kernel_size=15,
        down_channels=(72, 72, 72),
        down_kernel_size=(12, 15, 15),
        down_stride=(12, 9, 5),  # first element must be 12
        res_depth=3,
        res_kernel_size=15,
        se_ratio=4,
        out_kernel_size=21,
    )

    # Логгер для TensorBoard
    logger = TensorBoardLogger("lightning_logs", name="sleep_state_detection")

    # Колбэки

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",  # метрика для отслеживания
        dirpath="checkpoints/",  # куда сохранять модель
        filename="best-checkpoint",  # имя файла
        save_top_k=1,  # сохранить только лучший чекпоинт
        mode="min",  # 'min' для loss, 'max' для accuracy
        save_weights_only=True,  # сохраняем только веса, а не весь объект
    )
    early_stop_callback = EarlyStopping(monitor="valid_loss", patience=5, mode="min")

    # Обучение
    trainer = Trainer(
        max_epochs=2,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator="auto",
        devices="auto",
    )
    trainer.fit(model, datamodule=data_module)
