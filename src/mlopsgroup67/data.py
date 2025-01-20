#!/usr/bin/env python3

import os
from datasets import load_dataset
import pandas as pd

def main():
    # 1. Load all splits of the IMDB dataset.
    #    - If you're using "stanfordnlp/imdb" and getting 'unsupervised' errors, 
    #      either remove the unsupervised split, or use ignore_verifications=True.
    full_dataset = load_dataset("stanfordnlp/imdb")  # or "stanfordnlp/imdb"
    
    # 2. Remove the unsupervised split if you don't need it.
    #full_dataset.pop("unsupervised", None)
    
    # 3. Create an output directory for pickle files.
    output_dir = "data/raw/imdb_pickle"
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. Convert each split to a pandas DataFrame and save as pickle.
    for split_name, dataset_split in full_dataset.items():
        print(f"Converting {split_name} split to pandas and saving as pickle...")
        
        # Convert to pandas
        df = dataset_split.to_pandas()
        
        # Save as .pkl
        pickle_path = os.path.join(output_dir, f"{split_name}.pkl")
        df.to_pickle(pickle_path)
        
        print(f"  -> Saved {split_name} split to {pickle_path}")

if __name__ == "__main__":
    main()
