from typing import Optional
from typing import Tuple

import attr


@attr.s(auto_attribs=True)
class SlotCutParams:
    lr: float = 0.0004
    batch_size: int = 64
    val_batch_size: int = 64
    resolution: Tuple[int, int] = (128, 128)
    clevrtex_variant: str = "full"
    num_slots: int = 12
    data_root: str = "/local-ssd/aapervez"
    gpus: int = 1
    max_epochs: int = 500
    num_sanity_val_steps: int = 1
    scheduler_gamma: float = 0.8
    #scheduler_gamma: float = 1.0
    weight_decay: float = 0.0
    num_train_images: Optional[int] = None
    num_val_images: Optional[int] = None
    empty_cache: bool = True
    is_logger_enabled: bool = True
    is_verbose: bool = True
    num_workers: int = 4
    n_samples: int = 5
    warmup_steps_pct: float = 0.00
    decay_steps_pct: float = 0.2


    resume: bool = True
