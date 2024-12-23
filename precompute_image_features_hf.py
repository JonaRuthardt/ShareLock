import os
import torch
import json
import tqdm
import argparse
from PIL import Image

from datasets import load_dataset, Image
from torch.utils.data import DataLoader

from featureutils.core import FeatureUtils
from sharelock.models.vision_encoder import VisionEncoder
from sharelock.utils.misc import get_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Precompute features")
    parser.add_argument("--template", type=str, help="Template to use for class names", default="A photo of a {class_name}")
    parser.add_argument("--dataset", type=str, help="Dataset to precompute features for")
    parser.add_argument("--split", type=str, default="train", help="Split to precompute features for")
    parser.add_argument("--vision_model", type=str, help="Vision model to use")
    parser.add_argument("--output_dir", type=str, help="Directory to save the features", required=True)
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("TMPDIR", None), help="Directory to cache the model")
    parser.add_argument("--dataset_dir", type=str, default="datasets", help="Directory where the dataset is stored")
    parser.add_argument("--gpu_num", type=int, default=1, help="Number of GPUs being used")
    parser.add_argument("--gpu_id", type=int, default=0, help="ID of the GPU being used")
    args = parser.parse_args()
    
    assert args.dataset is not None, "Dataset must be provided"
    assert args.vision_model is not None, "Vision model must be provided"
    
    print(f"Precomputing features for dataset {args.dataset} using vision model {args.vision_model}")
    print(f"Computing features with {args.gpu_num} GPUs, starting at GPU {args.gpu_id}")
    
    # Initialize the feature storage util
    output_dir = f"{args.output_dir}/{args.dataset.split('/')[-1]}/{args.vision_model.split('/')[-1]}"
    feature_utils = FeatureUtils(base_dir=output_dir, staging_dir=args.cache_dir, feature_num=2)

    # Load the dataset
    dataset = load_dataset(args.dataset, split=args.split, trust_remote_code=True)
    dataset = dataset.cast_column("image", Image(mode="RGB"))
    transforms = get_transforms()
    def transform_fn(data):
        data['image'] = [transforms(img) for img in data['image']]
        return data
    dataset = dataset.with_transform(transform_fn)
    
    # Load the vision model
    model = VisionEncoder(args.vision_model).to(device)
    
    try:
        class_names = {idx: args.template.format(class_name=name.split(",")[0].strip()) for idx, name in enumerate(dataset.features["label"].names)}
        with open(f"{args.dataset_dir}/{args.dataset.split('/')[-1]}/class_names.json", "w", encoding="utf-8") as f:
            json.dump(class_names, f)
    except Exception as exc:
        raise ValueError("Selected dataset does not support currently implemented method of extracting class names") from exc
    
    # Setup the dataloader
    def collate_fn(hf_batch):
        images = torch.stack([torch.tensor(hf_batch[i]["image"]) for i in range(len(hf_batch))])
        labels = torch.tensor([hf_batch[i]["label"] for i in range(len(hf_batch))])
        return {"images": images, "labels": labels}
    
    dataloader = DataLoader(dataset, batch_size=64, num_workers=16, collate_fn=collate_fn)

    # Precompute the features
    image_id = 0
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        if batch_idx % args.gpu_num != args.gpu_id:
            image_id += len(batch["images"])
            continue
        features = model(batch["images"].to(device))
        for idx, image_id in enumerate(range(image_id, image_id + len(batch["images"]))):
            feature_utils.save_feature(image_id, vision_features=features[idx].detach().clone(), label=batch["labels"][idx].detach().clone())
        image_id += len(batch["images"])
            
    feature_utils.save()