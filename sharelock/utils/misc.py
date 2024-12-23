import torch
from torchvision import transforms
from transformers.models.clip.modeling_clip import clip_loss

feature_dimensions_vision = {
    "clip-ViT-B+32": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "vit_b_16": 768,
    "vit_b_32": 768,
    "dino_vitb16": 768,
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
    "dinov2_vits14_reg": 384,
    "dinov2_vitb14_reg": 768,
    "dinov2_vitl14_reg": 1024,
    "dinov2_vitg14_reg": 1536,
}

feature_dimensions_language = {
    "meta-llama/Meta-Llama-3-8B": 4096,
}

def loss(logits, target):
    logit_scale = torch.tensor(2.6592).exp()
    similarities = logit_scale * logits @ target.T
    return clip_loss(similarities)

def get_transforms(augmentation=False):
    if augmentation:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224, scale=(0.5,1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])        