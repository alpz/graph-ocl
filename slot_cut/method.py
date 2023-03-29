import pytorch_lightning as pl
import torch
from torch import optim
from torchvision import utils as vutils

from slot_cut.model import SlotCutModel
from slot_cut.params import SlotCutParams
from slot_cut.utils import Tensor
from slot_cut.utils import to_rgb_from_tensor
from slot_cut.clevertex import CLEVRTEX_Evaluator


class SlotCutMethod(pl.LightningModule):
    def __init__(self, model: SlotCutModel, datamodule: pl.LightningDataModule, params: SlotCutParams):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.params = params
        self.evaluator = CLEVRTEX_Evaluator()
        self.DEVICE = torch.device("cuda")

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        #print(type(batch))
        train_loss = self.model.loss_function(batch[1])
        logs = {key: val.item() for key, val in train_loss.items()}
        self.log_dict(logs, sync_dist=True)
        return train_loss

    def sample_images(self):
        dl = self.datamodule.val_dataloader()
        perm = torch.randperm(self.params.batch_size)
        idx = perm[: self.params.n_samples]
        batch = next(iter(dl))[1][idx]
        if self.params.gpus > 0:
            #batch = batch.to(self.device)
            batch = batch.to(self.DEVICE)
            model = self.model.to(self.DEVICE)
        recon_combined, recons, masks, slots = self.model.forward(batch)

        # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    batch.unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recons * masks + (1 - masks),  # each slot
                ],
                dim=1,
            )
        )

        batch_size, num_slots, C, H, W = recons.shape
        images = vutils.make_grid(
            out.view(batch_size * out.shape[1], C, H, W).cpu(), normalize=False, nrow=out.shape[1],
        )

        return images

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        val_loss = self.model.loss_function(batch[1])
        return val_loss


    def predict_step(self, batch, batch_idx, optimizer_idx=0):
        true_img = batch[1]
        true_mask = batch[2]
        pred_img, _, pred_mask, _ = self.model.forward(batch[1])
        #pred_img, pred_mask = self.eval(batch)

        #rescale to [0,1]
        pred_img = (pred_img+1)/2
        true_img = (true_img+1)/2

        self.evaluator.update(pred_img, pred_mask, true_img, true_mask)
        #return pred_img

    def on_predict_epoch_end(self, results):
        mse = self.evaluator.statistic('MSE')
        miou = self.evaluator.statistic('mIoU')
        ari = self.evaluator.statistic('ARI')
        ari_fg = self.evaluator.statistic('ARI_FG')

        logs = {
            "mse": mse,
            "miou": miou,
            "ari": ari,
            "ari-fg": ari_fg,
        }
        #self.log_dict(logs, sync_dist=True)
        print("\nMetrics ", logs)

    def test_step(self, batch, batch_idx, optimizer_idx=0):
        test_loss = self.model.loss_function(batch[1])
        return test_loss

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {
            "avg_test_loss": avg_loss,
        }
        self.log_dict(logs, sync_dist=True)
        print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {
            "avg_val_loss": avg_loss,
        }
        self.log_dict(logs, sync_dist=True)
        print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()

    def predict_dataloader(self):
        return self.datamodule.test_dataloader()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)

        warmup_steps_pct = self.params.warmup_steps_pct
        decay_steps_pct = self.params.decay_steps_pct
        total_steps = self.params.max_epochs * len(self.datamodule.train_dataloader())

        def warm_and_decay_lr_scheduler(step: int):
            warmup_steps = warmup_steps_pct * total_steps
            decay_steps = decay_steps_pct * total_steps
            assert step < total_steps
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= self.params.scheduler_gamma ** (step / decay_steps)
            return factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step",}],
        )
