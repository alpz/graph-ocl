from typing import Optional

import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision import transforms

from slot_cut.data import CLEVRTEXDataModule

from slot_cut.method import SlotCutMethod
from slot_cut.model import SlotCutModel
from slot_cut.params import SlotCutParams
from slot_cut.utils import ImageLogCallback
from slot_cut.utils import rescale
import os
import subprocess
import sys


def main(params: Optional[SlotCutParams] = None):
    if params is None:
        params = SlotCutParams()


    assert params.num_slots > 1, "Must have at least 2 slots."

    logger_name = "slot-attention-clevr-tex"
    logger = pl_loggers.WandbLogger(project="slot-attention-clevr-tex", name=logger_name)


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

    method = SlotCutMethod(model=model, datamodule=clevr_datamodule, params=params)

    trainer = Trainer(
        logger=logger if params.is_logger_enabled else False,
        accelerator="ddp" if params.gpus > 1 else None,
        num_sanity_val_steps=params.num_sanity_val_steps,
        gpus=params.gpus,
        max_epochs=params.max_epochs,
        log_every_n_steps=50,
        callbacks=[LearningRateMonitor("step"), ImageLogCallback(),] if params.is_logger_enabled else [],
    )
    trainer.fit(method)


#def write_source_files(summ_dir):
#
#    # write git diff, commit hash, redirect stdout
#    diff = os.path.join(summ_dir, 'git.diff')
#
#    if not os.path.isfile(diff):
#        with open(diff, 'w') as fd:
#            subprocess.call(['git diff'], stdout=fd, stderr=fd, shell=True)
#
#    # write commit hash
#    commit = os.path.join(summ_dir, 'commit.txt')
#    if not os.path.isfile(commit):
#        with open(commit, 'w') as fd:
#            subprocess.call(['git rev-parse HEAD'], stdout=fd, stderr=fd, shell=True)

if __name__ == "__main__":
    main()
