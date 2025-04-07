#!/usr/bin/env python3
"""
Generate synthetic receipt dataset for training.

This script generates a dataset of synthetic receipt images and creates
appropriate train/val/test splits for model training.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from utils.reproducibility import set_seed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate receipt dataset for training")
    parser.add_argument("--output_dir", default="datasets", help="Base output directory")
    parser.add_argument("--num_collages", type=int, default=300, help="Number of collages to generate")
    parser.add_argument("--count_probs", default="0.3,0.3,0.2,0.1,0.1,0", 
                      help="Probability distribution for receipt counts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--image_size", type=int, default=448, 
                      help="Output image size (default: 448 for InternVL2)")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Proportion for training set")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Proportion for validation set")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Proportion for test set")
    return parser.parse_args()


def generate_dataset(args):
    """
    Generate the synthetic receipt dataset.
    """
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Set the random seed
    set_seed(args.seed)
    
    # 2. Create the synthetic receipts
    synthetic_dir = output_dir / "synthetic_receipts"
    cmd = [
        "python", "data/data_generators/create_synthetic_receipts.py",
        "--num_collages", str(args.num_collages),
        "--count_probs", args.count_probs,
        "--output_dir", str(synthetic_dir),
        "--image_size", str(args.image_size),
        "--seed", str(args.seed)
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # 3. Create the dataset from synthetic collages
    dataset_dir = output_dir / "receipt_dataset"
    cmd = [
        "python", "data/data_generators/create_collage_dataset.py",
        "--collage_dir", str(synthetic_dir),
        "--output_dir", str(dataset_dir),
        "--image_size", str(args.image_size),
        "--train_ratio", str(args.train_ratio),
        "--val_ratio", str(args.val_ratio),
        "--test_ratio", str(args.test_ratio),
        "--seed", str(args.seed)
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    print(f"Dataset generation complete! Created images at {args.image_size}×{args.image_size} resolution")
    print(f"Dataset splits saved to {dataset_dir}")


if __name__ == "__main__":
    args = parse_args()
    
    # Validate split ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-10:
        print("Error: Split ratios must sum to 1.0")
        sys.exit(1)
    
    generate_dataset(args)