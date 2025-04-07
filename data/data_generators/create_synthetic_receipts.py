#!/usr/bin/env python3
"""
Generate synthetic receipt images for training.

This script creates synthetic receipt images with varying formats, content, and appearances
to serve as training data for the receipt counter model.
"""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from data.data_generators.receipt_processor import create_blank_image


def create_receipt_image(width=1000, height=2000, items_count=None, max_items=20):
    """
    Create a single synthetic receipt image.
    
    Args:
        width: Width of the receipt
        height: Height of the receipt
        items_count: Optional fixed number of items on receipt
        max_items: Maximum number of items if items_count is None
        
    Returns:
        PIL Image of the synthetic receipt
    """
    # Create receipt with original dimensions
    receipt = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(receipt)
    
    # Try to load a font, fallback to default if not available
    try:
        font_header = ImageFont.truetype("Arial", 24)
        font_body = ImageFont.truetype("Arial", 18)
    except IOError:
        font_header = ImageFont.load_default()
        font_body = ImageFont.load_default()
    
    # Add store name and header
    store_names = ["GROCERY STORE", "SUPERMARKET", "FOOD MART", "MARKET PLACE", "CONVENIENCE STORE"]
    store_name = random.choice(store_names)
    draw.text((width // 2 - 80, 50), store_name, fill="black", font=font_header)
    draw.text((width // 2 - 100, 80), "RECEIPT", fill="black", font=font_header)
    
    # Add date and time
    date = f"{random.randint(1, 12)}/{random.randint(1, 28)}/{random.randint(2020, 2023)}"
    time = f"{random.randint(0, 23)}:{random.randint(0, 59)}"
    draw.text((50, 130), f"Date: {date}", fill="black", font=font_body)
    draw.text((width - 200, 130), f"Time: {time}", fill="black", font=font_body)
    
    # Add receipt items
    y_pos = 200
    items = ["Milk", "Bread", "Eggs", "Cheese", "Apples", "Bananas", "Chicken", "Rice",
             "Pasta", "Cereal", "Coffee", "Tea", "Chocolate", "Yogurt", "Juice"]
    
    # Determine number of items
    if items_count is None:
        items_count = random.randint(5, max_items)
    
    total_amount = 0
    
    # Draw items, prices and total
    for i in range(items_count):
        if y_pos > height - 300:  # Ensure we have space for footer
            break
            
        item = random.choice(items)
        price = round(random.uniform(1.00, 20.00), 2)
        total_amount += price
        
        draw.text((50, y_pos), f"{item}", fill="black", font=font_body)
        draw.text((width - 150, y_pos), f"${price:.2f}", fill="black", font=font_body)
        y_pos += 30
    
    # Add separator line
    draw.line([(50, y_pos + 10), (width - 50, y_pos + 10)], fill="black", width=2)
    
    # Add tax and total
    tax = round(total_amount * 0.08, 2)  # 8% tax
    final_total = total_amount + tax
    
    y_pos += 30
    draw.text((50, y_pos), "Subtotal:", fill="black", font=font_body)
    draw.text((width - 150, y_pos), f"${total_amount:.2f}", fill="black", font=font_body)
    
    y_pos += 30
    draw.text((50, y_pos), "Tax:", fill="black", font=font_body)
    draw.text((width - 150, y_pos), f"${tax:.2f}", fill="black", font=font_body)
    
    y_pos += 30
    draw.text((50, y_pos), "Total:", fill="black", font=font_header)
    draw.text((width - 150, y_pos), f"${final_total:.2f}", fill="black", font=font_header)
    
    # Add footer
    y_pos += 60
    draw.text((width // 2 - 100, y_pos), "Thank you for shopping!", fill="black", font=font_body)
    
    # Add some random noise
    if random.random() > 0.5:
        receipt = receipt.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Add slight rotation for realism
    rotation = random.uniform(-5, 5)
    receipt = receipt.rotate(rotation, expand=True, fillcolor='white')
    
    return receipt


def create_receipt_collage(receipt_count, image_size=448):
    """
    Create a collage with a specified number of receipts.
    
    Args:
        receipt_count: Number of receipts to include (0-5)
        image_size: Size of the output collage image
        
    Returns:
        PIL Image containing the receipt collage
    """
    # Create blank image for collage
    collage = create_blank_image(image_size, image_size, 'white')
    
    # If no receipts, return blank image with some noise
    if receipt_count == 0:
        # Add some random noise/texture to avoid completely blank images
        draw = ImageDraw.Draw(collage)
        for _ in range(100):
            x1 = random.randint(0, image_size - 1)
            y1 = random.randint(0, image_size - 1)
            x2 = min(x1 + random.randint(1, 10), image_size - 1)
            y2 = min(y1 + random.randint(1, 10), image_size - 1)
            color = (random.randint(240, 255), random.randint(240, 255), random.randint(240, 255))
            draw.rectangle([x1, y1, x2, y2], fill=color)
        return collage
    
    # Generate the requested number of receipts
    receipts = []
    for _ in range(receipt_count):
        # Create a receipt with random dimensions
        width = random.randint(300, 600)
        height = random.randint(800, 1500)
        items_count = random.randint(5, 15)
        receipt = create_receipt_image(width, height, items_count)
        
        # Resize receipt to fit in collage, maintaining aspect ratio
        scale_factor = min(image_size / 2 / receipt.width, image_size / 2 / receipt.height)
        new_width = int(receipt.width * scale_factor)
        new_height = int(receipt.height * scale_factor)
        receipt = receipt.resize((new_width, new_height), Image.LANCZOS)
        
        receipts.append(receipt)
    
    # Place receipts in collage
    for idx, receipt in enumerate(receipts):
        if receipt_count == 1:
            # Center the receipt
            x_pos = (image_size - receipt.width) // 2
            y_pos = (image_size - receipt.height) // 2
        else:
            # Distribute receipts across the image
            if idx % 2 == 0:  # Left side
                x_pos = random.randint(10, image_size // 2 - receipt.width - 10)
            else:  # Right side
                x_pos = random.randint(image_size // 2 + 10, image_size - receipt.width - 10)
                
            y_pos = random.randint(10, image_size - receipt.height - 10)
        
        # Paste the receipt onto the collage
        collage.paste(receipt, (x_pos, y_pos))
    
    # Apply final touch-ups
    if random.random() > 0.7:
        # Add slight blur occasionally
        collage = collage.filter(ImageFilter.GaussianBlur(radius=0.3))
    
    return collage


def generate_dataset(output_dir, num_collages=300, count_probs=None, image_size=448):
    """
    Generate a dataset of receipt collages with varying receipt counts.
    
    Args:
        output_dir: Directory to save the generated images
        num_collages: Number of collage images to generate
        count_probs: Probability distribution for number of receipts (0-5)
        image_size: Size of output images
        
    Returns:
        DataFrame with image filenames and receipt counts
    """
    import pandas as pd
    
    # Create output directories
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Default distribution if not provided
    if count_probs is None:
        count_probs = [0.3, 0.3, 0.2, 0.1, 0.1, 0.0]  # Probabilities for 0, 1, 2, 3, 4, 5 receipts
    
    # Normalize probabilities to sum to 1
    count_probs = np.array(count_probs)
    count_probs = count_probs / count_probs.sum()
    
    # Generate collages
    data = []
    
    for i in range(num_collages):
        # Determine number of receipts based on probability distribution
        receipt_count = np.random.choice(len(count_probs), p=count_probs)
        
        # Create collage
        collage = create_receipt_collage(receipt_count, image_size)
        
        # Save image
        filename = f"receipt_collage_{i:05d}.png"
        collage.save(images_dir / filename)
        
        # Add to dataset
        data.append({
            "filename": filename,
            "receipt_count": receipt_count
        })
        
        # Progress update
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_collages} collages")
    
    # Create and save metadata
    df = pd.DataFrame(data)
    df.to_csv(output_dir / "metadata.csv", index=False)
    
    print(f"Dataset generation complete: {num_collages} images")
    print(f"Distribution of receipt counts: {df['receipt_count'].value_counts().sort_index()}")
    
    return df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic receipt dataset")
    parser.add_argument("--output_dir", default="synthetic_receipts", help="Output directory")
    parser.add_argument("--num_collages", type=int, default=300, help="Number of collages to generate")
    parser.add_argument("--count_probs", default="0.3,0.3,0.2,0.1,0.1,0", 
                      help="Probability distribution for receipt counts (0,1,2,3,4,5)")
    parser.add_argument("--image_size", type=int, default=448,
                      help="Size of output images (default: 448 for InternVL2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Parse probability distribution
    count_probs = [float(p) for p in args.count_probs.split(',')]
    
    # Generate dataset
    df = generate_dataset(
        output_dir=args.output_dir,
        num_collages=args.num_collages,
        count_probs=count_probs,
        image_size=args.image_size
    )