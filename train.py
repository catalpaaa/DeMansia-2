from dataclasses import asdict

import pytorch_lightning as pl
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from lightning.pytorch import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader

from model import DeMansia_2
from model_config import DeMansia_2_tiny_config
from modules.data import create_token_label_dataset, create_token_label_loader
from modules.ema import EMA, EMAModelCheckpoint

imagenet_root = "datasets/ImageNet 1k"
token_label_root = "datasets/ImageNet 1k token label"

config = asdict(DeMansia_2_tiny_config())
model = DeMansia_2(**config)


class dataset(pl.LightningDataModule):
    def __init__(self, batch_size: int, num_workers: int):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.train_set = create_token_label_dataset(
            imagenet_root + "/train", token_label_root
        )
        self.valid_set = datasets.ImageFolder(
            imagenet_root + "/val",
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )

    def train_dataloader(self):
        return create_token_label_loader(
            self.train_set,
            input_size=config["img_size"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            use_prefetcher=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


data = dataset(batch_size=256, num_workers=12)

trainer = pl.Trainer(
    callbacks=[
        EMA(decay=0.9999),
        EMAModelCheckpoint(
            dirpath="models/",
            save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ],
    logger=pl_loggers.WandbLogger(project="DeMansia 2 Tiny", name="Pretrain"),
    precision="bf16-mixed",
    max_epochs=300,
    accumulate_grad_batches=4,
)

trainer.fit(model, data)
# trainer.fit(model, data, ckpt_path="ckpt to resume training")
