# Unitree G1 SONIC datamodule (added for LocoManiLab integration).
"""`BaseDataModule` hardcodes `JsonDataset`, so the sonic action space needs its own
datamodule. Everything except the dataset class is identical; select it from the
hydra data config with `data.type: SonicDataModule`.
"""
from copy import deepcopy

from lightning import LightningDataModule
from mmengine import Config, DATASETS
from torch.utils.data import DataLoader, DistributedSampler

from mibot.data.collate.custom_collate import CustomCollate
from mibot.data.datasets.sonic_json_dataset import SonicJsonDataset


@DATASETS.register_module()
class SonicDataModule(LightningDataModule):
    def __init__(self, params: Config) -> None:
        super().__init__()
        self.params: Config = params
        self.batch_size: int = params.train_datasets.get("batch_size", 16)
        self.collate_fn = CustomCollate()
        self.train_set = None

    def setup(self, stage=None) -> None:
        if stage in (None, "fit") and self.train_set is None:
            self.train_set = SonicJsonDataset(deepcopy(self.params))

    def train_dataloader(self) -> DataLoader:
        if self.train_set is None:
            self.setup("fit")
        sampler = DistributedSampler(self.train_set, shuffle=True, seed=42)
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=8,
            prefetch_factor=4,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            pin_memory=True,
        )
