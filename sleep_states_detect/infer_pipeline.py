import torch
from pytorch_lightning import Trainer

from sleep_states_detect.data_manage.dataset import SleepDataModule
from sleep_states_detect.metrics.find_peaks import predict_peaks
from sleep_states_detect.models.unet1d_lightning import UNet1dLightning
from sleep_states_detect.utils.utils import get_latest_checkpoint


def infer_main():
    # Загрузка данных
    data_module = SleepDataModule()
    data_module.setup("predict")

    # Загрузка модели
    model = UNet1dLightning.load_from_checkpoint(get_latest_checkpoint("checkpoints"))

    # Предсказание
    trainer = Trainer()
    predictions = trainer.predict(model, datamodule=data_module)

    # Постпроцессинг данных
    data = data_module.make_results(torch.cat(predictions, dim=0))
    data = predict_peaks(data)

    # Сохранение результата
    data.to_csv("data/result.csv")
