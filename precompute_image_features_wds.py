import os
import torch
from tqdm import tqdm
import argparse

import webdataset as wds
from datasets import load_dataset, Image
from torch.utils.data import DataLoader

from featureutils.core import FeatureUtils
from sharelock.models.vision_encoder import VisionEncoder
from sharelock.utils.misc import get_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Precompute features")
    parser.add_argument("--dataset", type=str, help="Dataset to precompute features for", required=True)
    parser.add_argument("--vision_model", type=str, help="Vision model to use", required=True)
    parser.add_argument("--output_dir", type=str, help="Directory to save the features", required=True)
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("TMPDIR", None), help="Directory to cache the model")
    parser.add_argument("--dataset_dir", type=str, help="Directory where the dataset is stored", required=True)
    parser.add_argument("--gpu_num", type=int, default=1, help="Number of GPUs being used")
    parser.add_argument("--gpu_id", type=int, default=0, help="ID of the GPU being used")
    args = parser.parse_args()
    
    print(f"Precomputing features for dataset {args.dataset} using vision model {args.vision_model}")
    print(f"Computing features with {args.gpu_num} GPUs, starting at GPU {args.gpu_id}")
    
    # Initialize the feature storage util
    output_dir = f"{args.output_dir}/{args.dataset.split('/')[-1]}/{args.vision_model.split('/')[-1]}"
    feature_utils = FeatureUtils(base_dir=output_dir, staging_dir=args.cache_dir, feature_num=1)
    existing_keys = {int(key): None for key in feature_utils.list_keys()}
    
    transforms = get_transforms()
    
    # Load the dataset
    # shards_path = args.dataset_dir + "/{00000..99999}.tar"
    shards_path = [args.dataset_dir + "/" + shard for shard in os.listdir(args.dataset_dir) if shard.endswith(".tar")]
    shards_path = sorted([args.dataset_dir + "/" + shard for shard in os.listdir(args.dataset_dir) if shard.endswith(".tar")])
    print(len(shards_path))
    # exit()

    # Create a WebDataset pipeline
    dataset = wds.WebDataset(shards_path).decode("pil").to_tuple("jpg", "__key__").map_tuple(transforms, int)
    
    # Load the vision model
    model = VisionEncoder(args.vision_model).to(device)
    
    # Wrap dataset in a DataLoader
    dataloader = DataLoader(dataset, batch_size=32, num_workers=16)
    
    # Iterate through the DataLoader
    for images, keys in tqdm(dataloader):
        # Filter out keys that already exist
        non_existing_indices = [idx for idx, key in enumerate(keys) if int(key.item()) not in existing_keys]
        if not non_existing_indices:
            continue
        
        non_existing_images = images[non_existing_indices]
        non_existing_keys = [keys[idx] for idx in non_existing_indices]
        
        features = model(non_existing_images.to(device))
        for idx, key in enumerate(non_existing_keys):
            feature_utils.save_feature(int(key.item()), vision_features=features[idx].detach().clone())
    
    feature_utils.save()