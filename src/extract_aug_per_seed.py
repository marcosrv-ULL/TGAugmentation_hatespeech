#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract per-seed augmentations from a telephone-game style .txt log and
APPEND them to a JSONL dataset.

Features:
1. Copies all original entries from input_jsonl to output_jsonl.
2. Detects the highest numeric ID in the original file to continue the sequence.
3. Parses the .txt file to extract:
   - The augmentation text.
   - The category (from [category] brackets).
4. Appends new rows with: id, text_masked, predicted_hate_category, is_augmentation.

Usage example:

    python extract_aug_per_seed.py \
        --aug_txt augmented_en_raw.txt \
        --input_jsonl en_dataset_train.jsonl \
        --output_jsonl en_dataset_train_plus_aug.jsonl \
        --text_field text_masked
"""

import argparse
import json
import re
from typing import List, Dict, Any

# Structure to hold a batch of augmentations belonging to a specific category
class AugmentationBlock:
    def __init__(self, category: str):
        self.category = category
        self.augmentations = []

def parse_augmented_txt(path: str) -> List[AugmentationBlock]:
    """
    Parse the .txt augmentation log.
    Extracts category from lines like: "[sexual_orientation] seed id=None..."
    Collects memory steps associated with that category.
    """
    blocks: List[AugmentationBlock] = []
    
    current_category = None
    current_aug_list = []

    # Regex to capture content inside square brackets at the start of the line
    # Matches: [sexual_orientation] -> sexual_orientation
    category_pattern = re.compile(r"^\[(.*?)\]")

    def flush_block():
        nonlocal current_category, current_aug_list
        if current_category and current_aug_list:
            block = AugmentationBlock(current_category)
            block.augmentations = current_aug_list[:]
            blocks.append(block)
        current_aug_list = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            # 1. Check for Category Header line (e.g. [sexual_orientation] ...)
            match = category_pattern.match(line)
            if match:
                # If we were processing a previous block, flush it
                flush_block()
                
                # Set new category
                current_category = match.group(1)
                continue

            # 2. Augmented lines (memory step=...)
            if "memory step=" in line and ":" in line:
                if current_category is None:
                    # If for some reason we found a memory step before a category, skip or handle
                    continue
                    
                # split at the first ':' to drop the prefix "memory step=X (call=Y):"
                _, mem_text = line.split(":", 1)
                mem_text = mem_text.strip()
                if mem_text:
                    current_aug_list.append(mem_text)
                continue

            # Note: We ignore "seed_masked:" lines as we don't need the original text 
            # and the category is already captured in the header line.

        # Flush the final block after loop ends
        flush_block()

    return blocks


def append_augmentations_to_jsonl(
    aug_blocks: List[AugmentationBlock],
    input_jsonl: str,
    output_jsonl: str,
    text_field: str = "text_masked"
) -> None:
    """
    1. Copies lines from input_jsonl to output_jsonl.
    2. Calculates max_id from original data to generate new IDs.
    3. Appends new lines for every augmentation found.
    """
    original_count = 0
    new_aug_count = 0
    
    # We will try to find the max integer ID to continue numbering
    max_id = 0
    
    print("Copying original entries and calculating IDs...")
    
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        # 1. Copy original dataset
        with open(input_jsonl, "r", encoding="utf-8") as fin:
            for raw_line in fin:
                line = raw_line.strip()
                if not line:
                    continue
                
                try:
                    obj = json.loads(line)
                    # Try to track max ID
                    if "id" in obj:
                        try:
                            val = int(obj["id"])
                            if val > max_id:
                                max_id = val
                        except (ValueError, TypeError):
                            # ID is not an integer (maybe UUID or string), ignore for max calculation
                            pass
                except json.JSONDecodeError:
                    pass

                fout.write(line + "\n")
                original_count += 1
        
        # If no numeric IDs were found, we might want to start from the count + 1
        # or keep max_id as 0 if the dataset is 0-indexed.
        # Let's assume if max_id is 0 and we read rows, maybe IDs weren't numeric.
        # To be safe, let's set current_id_counter to max_id + 1
        current_id_counter = max_id + 1
        
        # 2. Append new augmentations
        print(f"Appending new augmentations starting from ID {current_id_counter}...")
        
        for block in aug_blocks:
            category = block.category
            for aug_text in block.augmentations:
                
                new_obj = {
                    "id": current_id_counter,
                    text_field: aug_text,
                    "predicted_hate_category": category,
                    "is_augmentation": True
                }
                
                fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
                
                current_id_counter += 1
                new_aug_count += 1

    print("-" * 40)
    print(f"Original entries copied: {original_count}")
    print(f"New augmented entries appended: {new_aug_count}")
    print(f"Total lines in output: {original_count + new_aug_count}")
    print(f"Last ID generated: {current_id_counter - 1}")
    print(f"Output written to: {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(
        description="Append augmentations from a telephone-game log "
                    "to the end of a JSONL dataset with new IDs."
    )
    parser.add_argument(
        "--aug_txt",
        required=True,
        help="Path to augmented_en_raw.txt (telephone-game log).",
    )
    parser.add_argument(
        "--input_jsonl",
        required=True,
        help="Path to the original JSONL dataset.",
    )
    parser.add_argument(
        "--output_jsonl",
        required=True,
        help="Path for the output JSONL.",
    )
    parser.add_argument(
        "--text_field",
        default="text_masked",
        help="Name of the field in JSONL to store the text content.",
    )

    args = parser.parse_args()

    print(f"Parsing augmentations from: {args.aug_txt}")
    aug_blocks = parse_augmented_txt(args.aug_txt)
    
    # Calculate total augmentations just for info
    total_augs = sum(len(b.augmentations) for b in aug_blocks)
    print(f"Found {len(aug_blocks)} category blocks with a total of {total_augs} augmentations.")

    append_augmentations_to_jsonl(
        aug_blocks=aug_blocks,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        text_field=args.text_field,
    )


if __name__ == "__main__":
    main()