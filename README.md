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

```bash
# Clone the repository
git clone https://github.com/tmnestor/internvl-receipt-counter.git
cd internvl-receipt-counter

# Install dependencies
pip install -e .
```

## Usage

### Data Generation

Generate synthetic receipt training data:

```bash
# Using the script directly (navigate to the script directory first)
cd data/data_generators/
python create_synthetic_receipts.py --output_dir ../../datasets/synthetic_receipts --num_collages 1000 --count_probs "0.3,0.3,0.2,0.1,0.1" --stapled_ratio 0.3

# Or using the convenience script (which also creates train/val/test splits)
python scripts/generate_data.py --output_dir datasets --num_collages 1000 --count_probs "0.3,0.3,0.2,0.1,0.1" --stapled_ratio 0.3
```

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