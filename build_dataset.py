import os
import pandas as pd
from protein_grid import ProteinGrid

def build_ml_dataset(pdb_folder, output_csv="pocket_dataset.csv"):
    """
    Loops through a folder of PDB files, extracts ML features for all pockets,
    and saves them into a single Pandas DataFrame.
    """
    all_pockets_data = []
    
    # 1. Loop through every file in your dataset folder
    for filename in os.listdir(pdb_folder):
        if filename.endswith(".pdb"):
            # Extract the protein name (e.g., "1STP" from "1STP.pdb")
            protein_id = filename.split(".")[0] 
            pdb_path = os.path.join(pdb_folder, filename)
            
            print(f"\n{'='*40}\nProcessing {protein_id}...\n{'='*40}")
            
            # 2. The Try-Except Block (Crucial for Data Science!)
            # If one PDB is corrupted or weird, it skips it instead of crashing the whole loop
            try:
                # Initialize your class
                protein = ProteinGrid(pdb_path)
                protein.scan_pockets()
                
                # Cluster them (Notice output_file=None, so it doesn't spam your hard drive with PDBs!)
                pockets = protein.cluster_and_export_pockets(output_file=None, min_size=50)
                
                # Analyze chemistry (Calculates your ML features)
                results = protein.analyze_all_pockets(pockets)
                
                # 3. Format the data for our ML spreadsheet
                for pocket_id, features in results.items():
                    # We add the protein ID and pocket ID so we know where this row came from
                    row = {
                        "protein_id": protein_id,
                        "pocket_id": pocket_id,
                        "volume_A3": features["volume_A3"],
                        "distance_to_core": features["distance_to_core"],
                        "hydrophobic_count": features["hydrophobic_count"],
                        "polar_count": features["polar_count"],
                        "positive_count": features["positive_count"],
                        "negative_count": features["negative_count"],
                        "aromatic_count": features["aromatic_count"],
                        "total_residues": features["total_residues_touching"]
                    }
                    all_pockets_data.append(row)
                    
            except Exception as e:
                print(f"Skipping {protein_id} due to an error: {e}")

    # 4. Convert the massive list into a single Pandas DataFrame and save as CSV
    print(f"\n{'='*40}\nFinished processing! Found {len(all_pockets_data)} total pockets.\n{'='*40}")
    
    if len(all_pockets_data) > 0:
        df = pd.DataFrame(all_pockets_data)
        df.to_csv(output_csv, index=False)
        print(f"Dataset successfully saved to {output_csv}")
    else:
        print("No pockets were found. Check your PDB files!")

if __name__ == "__main__":
    # IMPORTANT: Change this path to wherever you put your folder of PDB files!
    my_pdb_folder = "/home/diegovicente/pyt/pred_binding-site/pdb_dataset" 
    
    # Run the function
    build_ml_dataset(my_pdb_folder, output_csv="final_ml_dataset.csv")