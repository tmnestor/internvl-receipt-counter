# InternVL Receipt Counter

A receipt counting classification model built with InternVL-5-1B.

## Overview

This project implements a receipt counting classification model using the InternVL-5-1B vision-language model. It classifies images into three categories: 0 receipts, 1 receipt, or 2+ receipts.

The model leverages the powerful vision capabilities of InternVL with a 448×448 image resolution for optimal performance.

## Features

- High-resolution (448×448) image processing pipeline
- Memory-efficient 8-bit quantization support
- Three-stage training approach:
  1. Train only the classification head
  2. Fine-tune the vision encoder
  3. Fine-tune the complete model
- Comprehensive evaluation metrics
- Attention visualization for model interpretability

## Installation

### Method 1: Using pip (Simple data generation only)

```bash
# Clone the repository
git clone https://github.com/tmnestor/internvl-receipt-counter.git
cd internvl-receipt-counter

# Create and activate a virtual environment
python -m venv venv_py311
source venv_py311/bin/activate  # On Windows: venv_py311\Scripts\activate

# Install minimal dependencies (just for data generation)
pip install numpy pandas pillow matplotlib tqdm
```

### Method 2: Using conda (Full installation)

This method is recommended, especially on macOS where building `sentencepiece` can be problematic:

```bash
# Clone the repository
git clone https://github.com/tmnestor/internvl-receipt-counter.git
cd internvl-receipt-counter

# Create conda environment directly (safest method)
conda create -n internvl_env python=3.11 numpy pandas pillow matplotlib tqdm pytorch torchvision sentencepiece einops -c conda-forge -c pytorch

# Activate the environment (standard method)
conda activate internvl_env
```

Alternatively, you can use the environment.yml file:

```bash
# Create and activate conda environment from file
conda env create -f environment.yml
conda activate internvl_env
```

#### Activating in shared environments (Jupyter, cloud platforms)

In shared environments where you don't have admin access or want to avoid modifying `.bashrc`:

```bash
# Initialize conda without modifying .bashrc (recommended for shared systems)
source /opt/conda/etc/profile.d/conda.sh
conda activate internvl_env
```

If you encounter a "prefix already exists" error, use:

```bash
# Remove existing environment
conda env remove -n internvl_env

# Create the environment again
conda env create -f environment.yml
```

### Troubleshooting macOS Build Issues

If you're experiencing issues with `sentencepiece` installation on macOS:

1. Install the conda version (recommended):
   ```bash
   conda install -c conda-forge sentencepiece
   ```

2. Or, try installing with Homebrew:
   ```bash
   brew install sentencepiece
   ```

## Usage

### Data Generation

Generate high-resolution synthetic receipt training data (which will later be resized to 448x448 by the DataLoader):

```bash
# Generate 1000 high-resolution synthetic receipt images (2048x2048)
cd internvl-receipt-counter
python scripts/generate_data.py --output_dir datasets --num_collages 1000 --count_probs "0.3,0.3,0.2,0.1,0.1" --stapled_ratio 0.3 --image_size 2048
```

The images will be generated at high resolution (2048x2048) to simulate photos taken with a mobile phone camera. During training, the PyTorch DataLoader will automatically resize these images to 448x448 as required by the InternVL2 model.

#### About the Generated Data

The dataset includes different categories of images:

- **0 receipts**: Australian Taxation Office (ATO) documents rather than blank backgrounds
- **1 receipt**: A single receipt centered in the image
- **2+ receipts**: Multiple receipts arranged in the image
- **Stapled receipts**: For multi-receipt images, some may have stapled receipts (controlled by `stapled_ratio`)

The synthetic receipts simulate real-world receipt variations with:
- Different store names and layouts
- Varying items and prices
- Different receipt formats (standard, detailed, minimal, fancy)
- Natural variations like slight rotation, creases, and blur

You can control the distribution of receipt counts with the `count_probs` parameter, which accepts a comma-separated list of probabilities for 0, 1, 2, 3, 4, and 5 receipts.

### Training

Train the model:

```bash
# Using model path from config.yaml
python main.py --config config/config.yaml --mode train

# Override model path with command-line argument
python main.py --config config/config.yaml --mode train --model_path /path/to/pretrained/InternVL2_5-1B
```

### Evaluation

Evaluate the trained model:

```bash
# Using main.py
python main.py --config config/config.yaml --mode evaluate --model_path /path/to/trained/checkpoint.pt

# Or using the dedicated evaluation script
python scripts/evaluate.py --config config/config.yaml --model_path /path/to/trained/checkpoint.pt

# If you need to override the pretrained model path
python scripts/evaluate.py --model_path /path/to/trained/checkpoint.pt --pretrained_model_path /path/to/pretrained/InternVL2_5-1B
```

### Model Setup

Before running the application:

1. **REQUIRED**: Update the `config.yaml` file with the correct path to the pre-downloaded model:
   ```yaml
   model:
     pretrained_path: "/absolute/path/to/pretrained/InternVL2_5-1B"
   ```

2. Alternatively, you can provide the model path via command line:
   ```bash
   python main.py --model_path /absolute/path/to/pretrained/InternVL2_5-1B
   ```

**Note**: This application is designed to work in offline environments. The model must be pre-downloaded before running the application.

3. To pre-download the InternVL2 model using the provided utility script:
   ```bash
   # Download the model to a specific directory
   python utils/huggingface_model_download.py --model_name OpenGVLab/InternVL2_5-1B --output_dir /path/to/save/model
   
   # Example:
   python utils/huggingface_model_download.py --model_name OpenGVLab/InternVL2_5-1B --output_dir ~/models/InternVL2_5-1B
   ```
   
   After downloading, update the `pretrained_path` in your config.yaml to point to this location.

## Configuration

The project uses a YAML-based configuration system. The main configuration file is `config/config.yaml`, which includes:

- Data parameters
- Model architecture settings
- Training parameters
- Evaluation metrics

## Project Structure

```
internvl-receipt-counter/
├── config/              # Configuration files
├── data/                # Dataset implementation
│   └── data_generators/ # Synthetic data generation
│       ├── create_synthetic_receipts.py  # Generate synthetic receipt images
│       ├── create_receipt_collages.py    # Create collages from receipts
│       ├── create_collage_dataset.py     # Create train/val/test splits
│       └── receipt_processor.py          # Receipt image processing utilities
├── models/              # Model implementation
│   └── components/      # Model components
├── training/            # Training functionality
├── evaluation/          # Evaluation metrics and visualization
├── utils/               # Utility functions
└── scripts/             # Training and utility scripts
    ├── generate_data.py # Convenience script to generate data
    ├── train.py         # Training script
    └── evaluate.py      # Evaluation script
```

## License

MIT

## Acknowledgements

This project uses the InternVL architecture from:

"InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks"

