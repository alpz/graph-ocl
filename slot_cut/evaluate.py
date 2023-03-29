from typing import Optional

import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision import transforms

from slot_cut.data import CLEVRTEXDataModule
#from slot_cut.method import SlotAttentionMethod
#from slot_cut.model import SlotAttentionModel

from slot_cut.method import SlotCutMethod
from slot_cut.model import SlotCutModel
from slot_cut.params import SlotCutParams
from slot_cut.utils import ImageLogCallback
from slot_cut.utils import rescale
import os
import subprocess
import torch
import sys


def main(params: Optional[SlotCutParams] = None):
    if params is None:
        params = SlotCutParams()

    logger_name = "slot-attention-clevr-tex-eval"
    logger = pl_loggers.WandbLogger(project="slot-attention-clevr-tex-eval", name=logger_name)

    params.chk_path =sys.argv[1]
    print('read path' , params.chk_path)

    clevr_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(rescale),  # rescale between -1 and 1
            transforms.Resize(params.resolution),
        ]
    )

    clevr_datamodule = CLEVRTEXDataModule(
        data_root=params.data_root,
        clevrtex_variant=params.clevrtex_variant,
        max_n_objects=params.num_slots - 1,
        resolution=params.resolution,
        train_batch_size=params.batch_size,
        val_batch_size=params.val_batch_size,
        clevr_transforms=clevr_transforms,
        num_train_images=params.num_train_images,
        num_val_images=params.num_val_images,
        num_workers=params.num_workers,
    )

    model=SlotCutModel(
            resolution=params.resolution,
        num_slots=params.num_slots,
        empty_cache=params.empty_cache,
    )


    method = SlotCutMethod.load_from_checkpoint(params.chk_path, strict=False, model=model, datamodule=clevr_datamodule, params=params)

    trainer = Trainer(
        logger=False, #logger if params.is_logger_enabled else False,
        accelerator="ddp" if params.gpus > 1 else None,
        num_sanity_val_steps=params.num_sanity_val_steps,
        gpus=params.gpus,
        max_epochs=params.max_epochs,
        log_every_n_steps=50,
        #callbacks=[LearningRateMonitor("step"), ImageLogCallback(),] if params.is_logger_enabled else [],
    )
    trainer.predict(method)


if __name__ == "__main__":
    main()
