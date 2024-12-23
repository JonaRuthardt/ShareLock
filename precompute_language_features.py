import os
import torch
import json
import tqdm
import argparse

from featureutils.core import FeatureUtils
from sharelock.models.language_encoder import LanguageEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Precompute features")
    parser.add_argument("--dataset", type=str, help="Dataset to precompute features for")
    parser.add_argument("--language_model", type=str, help="Language model to use")
    parser.add_argument("--output_dir", type=str, help="Directory to save the features")
    parser.add_argument("--caption_file", type=str, default="captions.json", help="File containing the captions")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to cache the model")
    parser.add_argument("--model_cache_dir", type=str, default=os.environ.get("HF_HOME", None), help="Directory to cache the model")
    parser.add_argument("--dataset_dir", type=str, default="datasets", help="Directory where the dataset is stored")
    parser.add_argument("--gpu_num", type=int, default=1, help="Number of GPUs being used")
    parser.add_argument("--gpu_id", type=int, default=0, help="ID of the GPU being used")
    args = parser.parse_args()
    
    print(f"Precomputing features for dataset {args.dataset} using language model {args.language_model}")
    print(f"Computing features with {args.gpu_num} GPUs, starting at GPU {args.gpu_id}")
    
    if args.model_cache_dir is None and args.cache_dir is not None:
        args.model_cache_dir = args.cache_dir
    
    # Initialize the feature storage util
    output_dir = f"{args.output_dir}/{args.dataset.split('/')[-1]}/{args.language_model.split('/')[-1]}/{args.caption_file.replace('.json', '')}"
    feature_utils = FeatureUtils(base_dir=output_dir, staging_dir=args.cache_dir, feature_num=1)

    # Load the dataset
    with open(f"{args.dataset_dir}/{args.dataset.split('/')[-1]}/{args.caption_file}", "r", encoding="utf-8") as f:
        captions = json.load(f)

    # Load the language model
    model = LanguageEncoder(args.language_model, cache_dir=args.model_cache_dir).to(device)
    
    print(f"Number of existing features: {len(feature_utils.list_keys())}")
    
    # Precompute the features
    for image_idx, (image_id, caption) in tqdm.tqdm(enumerate(captions.items()), total=len(captions)):
        if image_idx % args.gpu_num != args.gpu_id or feature_utils.exists(image_id):
            continue
        
        features = model(caption)
        feature_utils.save_feature(image_id, language_features=features)
        
    feature_utils.save()