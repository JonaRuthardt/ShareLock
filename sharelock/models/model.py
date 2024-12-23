import torch
import torch.nn as nn
import pytorch_lightning as pl
from PIL import Image

from transformers import get_cosine_schedule_with_warmup

from sharelock.models.language_encoder import LanguageEncoder
from sharelock.models.vision_encoder import VisionEncoder
from sharelock.models.projection import Projection
from sharelock.utils.misc import loss, feature_dimensions_vision, feature_dimensions_language, get_transforms

class ShareLock(pl.LightningModule):
    def __init__(self, config):
        super(ShareLock, self).__init__()
        self.config = config.copy()
        
        self.save_hyperparameters(config)
        
        
        self.loss = loss
        
        # Setup of projection layers
        try:
            self.image_feature_size = feature_dimensions_vision[config.model.vision_encoder]
        except KeyError:
            raise ValueError(f"Feature dimensions for vision encoder {config.model.vision_encoder} not found. Please add it to the ‘feature_dimensions_vision‘ dictionary (sharelock/utils/misc.py).")
        try:
            self.language_feature_size = feature_dimensions_language[config.model.language_encoder]
        except KeyError:
            raise ValueError(f"Feature dimensions for language encoder {config.model.language_encoder} not found. Please add it to the ‘feature_dimensions_language‘ dictionary (sharelock/utils/misc.py).")
        if self.config.model.vision_projector.num_layers < 1:
            self.embedding_space_dim = self.image_feature_size
        elif self.config.model.language_projector.num_layers < 1:
            self.embedding_space_dim = self.language_feature_size
        else:
            self.embedding_space_dim = self.config.model.embedding_space_dim
        
        self.vision_projector = Projection(config.model.vision_projector, self.embedding_space_dim, input_size=self.image_feature_size)
        self.vision_encoder = None
        
        self.language_projector = Projection(config.model.language_projector, self.embedding_space_dim, input_size=self.language_feature_size)
        self.language_encoder = None
        
        self.class_prototypes = None
        
        
    def forward(self, batch: dict):
        image_features_projected = self.vision_projector(batch["vision_features"])
        # print(batch["language_features"].shape)
        # exit()
        langauge_features_projected = self.language_projector(batch["language_features"].squeeze())
        return image_features_projected, langauge_features_projected
    
    def training_step(self, batch: dict, batch_idx: int):
        image_features_projected, langauge_features_projected = self(batch)
        loss = self.loss(image_features_projected, langauge_features_projected)
        
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch: dict, batch_idx: int):
        image_features_projected, langauge_features_projected = self(batch)
        loss = self.loss(image_features_projected, langauge_features_projected)
        
        self.log(f"validation_loss", loss)
        return loss
    
        
    def test_step(self, batch: dict, batch_idx: int):
        if batch_idx == 0:
            self.setup_class_prototypes()
        
        image_features_projected = self.vision_projector(batch["vision_features"])
        predictions = self.predict(image_features_projected)
        accuracy = (predictions == batch["label"]).float().mean()
        
        self.log(f"test_accuracy", accuracy)
        return accuracy
        
    def predict(self, image_features_projected: torch.Tensor):
        logits = image_features_projected @ self.class_prototypes
        return torch.argmax(logits, dim=1)
    
    def setup_class_prototypes(self):
        dataset = self.trainer.datamodule.test_dataloader().dataset
        class_features = dataset.get_class_features().to(self.device)
        self.class_prototypes = self.language_projector(class_features).T
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.training.learning_rate, weight_decay=self.config.training.weight_decay)
        
        if self.config.training.lr_schedule == "constant":
            return optimizer
        lr_scheduler = {
                            "scheduler": get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.config.training.lr_warmup_steps, num_training_steps=self.config.training.max_steps),
                            "interval": "step",
                            "frequency": 1,
                        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    
    
    def encode_image(self, image: [torch.Tensor, Image.Image]):
        """Inference function to encode and project an image (preprocessed tensor or Image)"""
        
        if isinstance(image, Image.Image):
            image = get_transforms(self.config.model.vision_encoder)(image).unsqueeze(0)
            
        if self.vision_encoder is None:
            self.vision_encoder = VisionEncoder(self.config.model.vision_encoder).to(self.device)
            
        return self.vision_projector(self.vision_encoder(image.to(self.device)))
    
    def encode_text(self, text: str):
        """Inference function to encode and project a text (unprocessed raw-test)"""
        
        if self.language_encoder is None:
            self.language_encoder = LanguageEncoder(self.config.model.language_encoder).to(self.device)
            
        return self.language_projector(self.language_encoder(text))
    