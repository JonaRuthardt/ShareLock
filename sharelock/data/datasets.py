import os, sys
import torch
import random
from torch.utils.data import Dataset, Subset

from featureutils.core import FeatureUtils


class VisionLanguageFeatureDataset(Dataset):
    def __init__(self, config, split):
        self.config = config.copy()
        self.split = split
        
        data_staging_dir = os.environ.get("TMPDIR", None)
        rng = random.Random(self.config.seed)
                
        # Loading precomputed image features
        feature_dir = f"{self.config.data.precomputed_features_dir}/{self.config.data.dataset}/{self.config.model.vision_encoder.split('/')[-1]}"
        self.image_features = FeatureUtils(base_dir=feature_dir, staging_dir=data_staging_dir, require_features_exist=True)
        if len(self.image_features.list_keys()) == 0:
            raise ValueError(f"No vision features found in for dataset {self.config.data.dataset} and vision encoder {self.config.model.vision_encoder}")
        self.image_features.stage_data(features=["vision_features"])
        
        # Setup image ids and create splits
        feature_ids = self.image_features.list_keys()
        rng.shuffle(feature_ids)
        if split == "train":
            feature_ids = feature_ids[self.config.data.val_split_num:]
        elif split == "val":
            feature_ids = feature_ids[:self.config.data.val_split_num]
        self.idx2id = {idx: image_id for idx, image_id in enumerate(feature_ids)}
        
        # Loading precomputed language features for each caption file (randomly select caption at each iteration)
        self.language_features = []
        self.config.data.caption_files = [self.config.data.caption_files] if isinstance(self.config.data.caption_files, str) else self.config.data.caption_files
        for caption_file in self.config.data.caption_files:
            feature_dir = f"{self.config.data.precomputed_features_dir}/{self.config.data.dataset}/{self.config.model.language_encoder.split('/')[-1]}/{caption_file.replace('.json', '')}"
            language_features = FeatureUtils(base_dir=feature_dir, staging_dir=data_staging_dir, require_features_exist=True)
            if len(language_features.list_keys()) == 0:
                raise ValueError(f"No language features found in for dataset {self.config.data.dataset} and language encoder {self.config.model.language_encoder} and caption file {caption_file}")
            language_features.stage_data()
            self.language_features.append(language_features)
            
    def __len__(self):
        return len(self.idx2id)
    
    def __getitem__(self, idx):
        image_id = self.idx2id[idx]
        features = self.image_features.load_feature(image_id, ["vision_features"])
        features.update(random.choice(self.language_features).load_feature(image_id, ["language_features"]))
        features["language_features"] = features["language_features"].squeeze()
        return features
    
class ClassificationFeatureDataset(Dataset):
    def __init__(self, config):
        self.config = config.copy()
        
        data_staging_dir = os.environ.get("TMPDIR", None)
        
        feature_dir = f"{self.config.data.precomputed_features_dir}/{self.config.data.dataset}/{self.config.model.vision_encoder.split('/')[-1]}"
        self.image_features = FeatureUtils(base_dir=feature_dir, staging_dir=data_staging_dir, require_features_exist=True)
        if len(self.image_features.list_keys()) == 0:
            raise ValueError(f"No vision features found in for dataset {self.config.data.dataset} and vision encoder {self.config.model.vision_encoder}")
        self.image_features.stage_data()
        self.feature_idxs = self.image_features.list_keys()
        
        class_names_feature_dir = f"{self.config.data.precomputed_features_dir}/{self.config.data.dataset}/{self.config.model.language_encoder.split('/')[-1]}/{config.data.caption_files.replace('.json', '')}"
        self.class_names_features = FeatureUtils(base_dir=class_names_feature_dir, staging_dir=data_staging_dir, require_features_exist=True)
        if len(self.class_names_features.list_keys()) == 0:
            raise ValueError(f"No language features found in for dataset {self.config.data.dataset} and language encoder {self.config.model.language_encoder}")
        self.class_names_features.stage_data()
        
    def __len__(self):
        return len(self.image_features.list_keys())
    
    def __getitem__(self, idx):
        image_id = self.feature_idxs[idx]
        features = self.image_features.load_feature(image_id, ["vision_features", "label"])
        return features
    
    def get_class_features(self):
        class_features = []
        for class_id in self.class_names_features.list_keys():
            features = self.class_names_features.load_feature(class_id, ["language_features"])
            class_features.append(features["language_features"].squeeze())
        return torch.stack(class_features)