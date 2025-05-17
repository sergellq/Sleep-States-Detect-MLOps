import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from sleep_states_detect.metrics.compute_metric import score
from sleep_states_detect.metrics.find_peaks import predict_peaks
from sleep_states_detect.models.unet1d import UNet1d


class UNet1dLightning(pl.LightningModule):
    def __init__(
        self,
        input_channels,
        initial_channels,
        initial_kernel_size,
        down_channels,
        down_kernel_size,
        down_stride,
        res_depth,
        res_kernel_size,
        se_ratio,
        out_kernel_size,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Define the model from the UNet1d class
        self.model = UNet1d(
            input_channels=input_channels,
            initial_channels=initial_channels,
            initial_kernel_size=initial_kernel_size,
            down_channels=down_channels,
            down_kernel_size=down_kernel_size,
            down_stride=down_stride,
            res_depth=res_depth,
            res_kernel_size=res_kernel_size,
            se_ratio=se_ratio,
            out_kernel_size=out_kernel_size,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, Y, mask = batch
        preds = self(X) * mask
        loss = torch.nn.MSELoss()(preds, Y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y, mask = batch
        preds = self(X) * mask
        loss = torch.nn.MSELoss()(preds, Y)
        self.log("valid_loss", loss)

        # считаем среднее предсказание для графика
        if batch_idx == 0:
            self.preds = []
            self.sum_target = torch.zeros(Y[0].shape)
            self.num_target = 0
        self.preds += [preds.detach().cpu()]
        self.sum_target += Y.detach().sum(axis=0).cpu()
        self.num_target += Y.shape[0]

        return loss

    def predict_step(self, batch, batch_idx):
        X, _, mask = batch
        preds = self(X) * mask
        return preds

    def on_validation_epoch_end(self):
        # график распределения по всему дню
        mean_preds = torch.mean(torch.cat(self.preds), dim=0)
        mean_target = self.sum_target / self.num_target

        self.logger.experiment.add_figure(
            "Mean Prediction vs Target",
            self._plot_predictions(mean_preds, mean_target),
            self.current_epoch,
        )

        # целевая метрика
        data = self.trainer.datamodule.make_results(torch.cat(self.preds, dim=0))
        data = predict_peaks(data)
        score_all, df_score, df_result = score(
            solution=self.trainer.datamodule.val_df_events,
            submission=data,
        )
        self.log("target_metric", score_all)

    def _plot_predictions(self, mean_preds, mean_target):
        fig, ax = plt.subplots(figsize=(12, 3))

        # Assuming Y is the ground truth targets from dataset
        time_points = torch.linspace(0, 24, mean_preds.shape[0]).numpy()

        ax.plot(time_points, mean_preds, label="Mean of Prediction")
        ax.plot(time_points, mean_target, label="Mean of Target")

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)

        ax.set_xlim(0, 24)
        ax.set_xlabel("Hour (UTC)")
        ax.set_ylabel("1 = wake up; -1 = fall asleep")
        ax.legend()

        return fig

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=15)
        return [optimizer], [scheduler]
