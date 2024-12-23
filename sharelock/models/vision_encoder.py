import torch
import torchvision
import pytorch_lightning as pl

class VisionEncoder(pl.LightningModule):
    def __init__(self, model_name):
        super(VisionEncoder, self).__init__()
        
        self.model_name = model_name
                
        try:
            if "dinov2" in model_name:
                self.encoder = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
            elif "dino" in model_name:
                self.encoder = torch.hub.load('facebookresearch/dino', model_name, pretrained=True)
            elif "clip" in model_name:
                import clip
                self.encoder = clip.load(self.model_name.replace("clip-", "").replace("+", "/"))
            else:
                self.encoder = torchvision.models.__dict__[model_name](pretrained=True)
        except:
            raise ValueError(f"Vision model {model_name} not found or not implemented.")
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
    def forward(self, x):
        with torch.no_grad():
            if "clip" in self.model_name:
                return self.encoder.encode_image(x)
            return self.encoder(x)