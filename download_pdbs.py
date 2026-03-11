#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 13:07:17 2026

@author: pauvillen14
"""

import os
import urllib.request
import random
import concurrent.futures

def download_single_pdb(pdb_id, output_folder):
    """Helper function to download just one file."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    destination = os.path.join(output_folder, f"{pdb_id.upper()}.pdb")
    
    try:
        # Check if we already downloaded it to avoid duplicates
        if not os.path.exists(destination):
            urllib.request.urlretrieve(url, destination)
        return True
    except Exception:
        return False

def fast_batch_download(id_file, output_folder, max_downloads=1100):
    """Reads the IDs and downloads them simultaneously using multiple threads for faster execution."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the text file
    with open(id_file, "r") as file:
        raw_text = file.read()
        
    # Split by commas and clean up any accidental blank spaces
    pdb_ids = [code.strip().lower() for code in raw_text.split(',') if code.strip()]
    
    # Grab 1100 random IDs
    if len(pdb_ids) > max_downloads:
        pdb_ids = random.sample(pdb_ids, max_downloads)
        
    success_count = 0
    
    # max_workers=20 to speed up the process
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        
        # Submit all tasks to the thread pool
        futures = [executor.submit(download_single_pdb, pid, output_folder) for pid in pdb_ids]
        
        # As each file finishes downloading, update the count
        for future in concurrent.futures.as_completed(futures):
            if future.result():  
                success_count += 1
            
            # Print an update every 100 files 
            if success_count % 100 == 0:
                print(f"Fast Download Progress: {success_count} / {len(pdb_ids)}...")

    print(f"\nDone! Successfully blasted {success_count} files into {output_folder}")

if __name__ == "__main__":
    my_id_list = "/Users/pauvillen14/Desktop/BIOINFO/SBI/PROJECT/pred_binding-site/rcsb_pdb_ids_c51d65b716467352649bee98f12edd5e_00001-10000.txt" 
    
    my_dataset_folder = "/Users/pauvillen14/Desktop/BIOINFO/SBI/PROJECT/pred_binding-site/pdb_dataset" 
    
    fast_batch_download(my_id_list, my_dataset_folder, max_downloads=1100)