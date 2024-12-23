import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from sharelock.data.datasets import VisionLanguageFeatureDataset, ClassificationFeatureDataset

class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super(DataModule, self).__init__()
        
        self.config = config.copy()
        
        self.num_workers = self.config.data.num_workers
        self.batch_size = self.config.training.batch_size
                
    def setup(self, stage=None):        
        if stage == "fit" or stage is None:
            if isinstance(self.config.data.dataset, str):
                # Perform training and validation on single dataset
                self.train_dataset = VisionLanguageFeatureDataset(self.config, split="train")
                self.val_dataset = VisionLanguageFeatureDataset(self.config, split="val")
            else:
                # Perform training and validation on multiple datasets
                train_datasets = []
                val_datasets = []
                for dataset in self.config.data.dataset:
                    config = self.config.copy()
                    config.data.dataset = dataset
                    train_datasets.append(VisionLanguageFeatureDataset(config, split="train"))
                    val_datasets.append(VisionLanguageFeatureDataset(config, split="val"))
                self.train_dataset = torch.utils.data.ConcatDataset(train_datasets)
                self.val_dataset = torch.utils.data.ConcatDataset(val_datasets)
                    
            
        if stage == "test" or stage is None:
            # Perform test on ImageNet1k
            config = self.config.copy()
            config.data.dataset = "imagenet-1k"
            config.data.caption_files = "class_names.json"
            self.test_dataset = ClassificationFeatureDataset(config)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, prefetch_factor=12)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, prefetch_factor=12)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, prefetch_factor=12)