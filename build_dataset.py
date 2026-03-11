import os
import pandas as pd
from protein_grid import ProteinGrid

def build_ml_dataset(pdb_folder, output_csv="pocket_dataset.csv"):
    """
    Loops through a folder of PDB files, extracts ML features for all pockets,
    and saves them into a single Pandas DataFrame.
    """

    all_pockets_data = []
    
    # Loop through every file in the dataset folder
    for filename in os.listdir(pdb_folder):

        if filename.endswith(".pdb"):
            protein_id = filename.split(".")[0] 
            pdb_path = os.path.join(pdb_folder, filename)
            
            print(f"\n{'='*40}\nProcessing {protein_id}...\n{'='*40}")
            
            try:
                # Initialize the ProteinGrid and scan for pockets
                protein = ProteinGrid(pdb_path)
                protein.scan_pockets()
                
                # Cluster distinct pockets
                pockets = protein.cluster_and_export_pockets(output_file=None, min_size=50)
                
                # Analyze their chemistry (calculate ML features)
                results = protein.analyze_all_pockets(pockets)
                
                # Format the data for the ML spreadsheet
                for pocket_id, features in results.items():
                    # Add the protein ID and pocket ID so we know where this row came from
                    row = {
                        "protein_id": protein_id,
                        "pocket_id": pocket_id,
                    }

                    # Add all features EXCEPT lists and redundant raw counts
                    features_to_skip = [
                        "residue_list", "hydrophobic_count", "polar_count", 
                        "positive_count", "negative_count", "aromatic_count"
                    ]
                    
                    for key, value in features.items():
                        if key not in features_to_skip:
                            row[key] = value
                    
                    all_pockets_data.append(row)
                    
            except Exception as e:
                print(f"Skipping {protein_id} due to an error: {e}")

    # Convert the massive all_pockets_data list into a single Pandas DataFrame and save as CSV
    print(f"\n{'='*40}\nFinished processing! Found {len(all_pockets_data)} total pockets.\n{'='*40}")
    
    if len(all_pockets_data) > 0:
        df = pd.DataFrame(all_pockets_data)
        df.to_csv(output_csv, index=False)
        print(f"Dataset successfully saved to {output_csv}")
    else:
        print("No pockets were found. Check your PDB files!")

if __name__ == "__main__":
    # IMPORTANT: Change this path to wherever you put your folder of PDB files!
    my_pdb_folder = "/Users/pauvillen14/Desktop/BIOINFO/SBI/PROJECT/pred_binding-site/pdb_dataset" 
    
    # Run the building function
    build_ml_dataset(my_pdb_folder, output_csv="final_ml_dataset.csv")