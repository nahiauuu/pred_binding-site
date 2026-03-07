import numpy as np
from Bio.PDB import PDBParser
import scipy.ndimage as ndi
from scipy.spatial import KDTree

class ProteinGrid:

    HYDROPHOBIC = {'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TRP', 'MET'}
    POLAR = {'SER', 'THR', 'CYS', 'ASN', 'GLN', 'TYR'}
    AROMATIC = {'PHE', 'TYR', 'TRP', 'HIS'}
    POSITIVE = {'ARG', 'LYS', 'HIS'}
    NEGATIVE = {'ASP', 'GLU'}

    def __init__(self, pdb_file, grid_spacing=1.0, padding=3.0):
        """
        Initialize the class with the PDB file, the size of each voxel (grid_spacing),
        and the extra margin around the protein (padding).
        """

        self.pdb_file = pdb_file
        self.grid_spacing = grid_spacing
        self.padding = padding
        self.atoms_coords = []
        self.atom_metadata = []
        self.grid = None
        self.grid_shape = None
        
        # Execute initial setup methods
        self.extract_coordinates()
        self.calculate_bounding_box()
        self.map_to_grid()

    def extract_coordinates(self):
        """
        Use Biopython to extract the X, Y, Z coordinates of all atoms in the protein.
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", self.pdb_file)
        
        # Iterate through the model, chains, residues, and atoms
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Skip water molecules and heteroatoms (ligands)
                    # Biopython marks standard amino acids with ' '
                    if residue.get_id()[0] != ' ':
                        continue

                    for atom in residue:
                        self.atoms_coords.append(atom.coord)

                        atom_info = {
                            'atom_name': atom.get_name(),       
                            'residue_name': residue.get_resname(), 
                            'residue_id': residue.get_id()[1],     
                            'chain_id': chain.get_id()          
                        }
                        self.atom_metadata.append(atom_info)
                        
        self.atoms_coords = np.array(self.atoms_coords)
        print(f"Extracted {len(self.atoms_coords)} atoms.")

    def calculate_bounding_box(self):
        """
        Calculate the boundaries of the 3D bounding box that encloses the protein.
        """
        # Find the minimum and maximum X, Y, Z values among all atoms
        self.min_coords = np.min(self.atoms_coords, axis=0) - self.padding
        max_coords = np.max(self.atoms_coords, axis=0) + self.padding
        
        # Calculate the overall dimensions of the bounding box
        box_dimensions = max_coords - self.min_coords
        print(f"Bounding box dimensions: {box_dimensions}")
        
        # Determine how many 3D pixels (voxels) fit within the bounding box
        self.grid_shape = np.ceil(box_dimensions / self.grid_spacing).astype(int)
        print(f"Grid shape (X, Y, Z): {self.grid_shape}")
        
        # Create an empty 3D array (filled with zeros) using NumPy
        self.grid = np.zeros(self.grid_shape, dtype=int)
    
    def map_to_grid(self):
        """
        Maps the continuous 3D coordinates into the discrete 3D grid.
        Assigns a value of 1 to voxels containing at least one atom.
        """
        for coord in self.atoms_coords:
            # Shift the coordinates so the minimum bounding box is at (0,0,0)
            shifted_coord = coord - self.min_coords
            
            # Divide by grid_spacing to find the corresponding matrix index
            grid_index = (shifted_coord / self.grid_spacing).astype(int)
            
            # Extract X, Y, Z matrix indices
            ix, iy, iz = grid_index
            
            # Mark the voxel as occupied (Protein = 1)
            self.grid[ix, iy, iz] = 1
            
        occupied_voxels = np.sum(self.grid)
        print(f"Success: {len(self.atoms_coords)} atoms mapped into {occupied_voxels} out of {self.grid.size} voxels.")

    def scan_pockets(self):
        """
        Scans the 3D grid along the X, Y, and Z axes to find empty spaces (0s)
        that are trapped between protein atoms (1s) on both sides.
        """
        print("\nStarting LIGSITE 3-axis scan...")
        
        # 1. Sweep along the X-axis (Left and Right)
        # We check if there is a '1' anywhere to the left, and anywhere to the right
        left_sweep = np.maximum.accumulate(self.grid, axis=0)
        right_sweep = np.maximum.accumulate(self.grid[::-1, :, :], axis=0)[::-1, :, :]
        bounded_x = (left_sweep == 1) & (right_sweep == 1)
        
        # 2. Sweep along the Y-axis (Top and Bottom)
        top_sweep = np.maximum.accumulate(self.grid, axis=1)
        bottom_sweep = np.maximum.accumulate(self.grid[:, ::-1, :], axis=1)[:, ::-1, :]
        bounded_y = (top_sweep == 1) & (bottom_sweep == 1)
        
        # 3. Sweep along the Z-axis (Front and Back)
        front_sweep = np.maximum.accumulate(self.grid, axis=2)
        back_sweep = np.maximum.accumulate(self.grid[:, :, ::-1], axis=2)[:, :, ::-1]
        bounded_z = (front_sweep == 1) & (back_sweep == 1)
        
        # ... (keep the X, Y, Z sweep code from before) ...
        
        # 4. Count the number of trapping axes (PSP events)
        # We convert booleans (True/False) to 1s and 0s and add them up
        psp_count = bounded_x.astype(int) + bounded_y.astype(int) + bounded_z.astype(int)
        
        # 5. Identify the true surface pockets!
        is_empty = (self.grid == 0)
        
        # A good surface pocket is trapped in AT LEAST 2 directions.
        # (If you only want buried cavities, you'd use psp_count == 3)
        self.pocket_grid = is_empty & (psp_count >= 2)
        
        pocket_volume = np.sum(self.pocket_grid)
        print(f"Scan complete! Found {pocket_volume} pocket voxels.")

    def cluster_and_export_pockets(self, output_file=None, min_size=50):
        """
        Groups adjacent pocket voxels into distinct pockets, filters out tiny ones,
        and saves them as dummy atoms in a PDB file for PyMOL visualization.
        """
        print(f"\nClustering pockets and filtering out those smaller than {min_size} voxels...")

        # 1. Group connected voxels. 'labels' is a new grid where Pocket 1 is filled 
        # with 1s, Pocket 2 with 2s, etc. 'num_features' is the total number of pockets.
        labels, num_features = ndi.label(self.pocket_grid)
        
        valid_pockets = 0
        extracted_pockets = {}

        # Loop through each discovered pocket
        for pocket_id in range(1, num_features + 1):
            # Find all voxel indices belonging to this specific pocket
            voxel_indices = np.argwhere(labels == pocket_id)
            
            # Filter out small noise clusters
            if len(voxel_indices) < min_size:
                continue
                
            valid_pockets += 1
            extracted_pockets[valid_pockets] = voxel_indices

            print(f"Pocket {valid_pockets}: {len(voxel_indices)} voxels")
        
        # Only write the PDB file if an output_file string was actually provided!
        if output_file:
            with open(output_file, "w") as f:
                atom_serial = 1
                for valid_id, voxel_indices in extracted_pockets.items():
                    for index in voxel_indices:
                        real_coord = (index * self.grid_spacing) + self.min_coords
                        x, y, z = real_coord
                        f.write(
                            f"HETATM{atom_serial:5d}  POC STP A{valid_id:4d}    "
                            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n"
                        )
                        atom_serial += 1
            print(f"Saved visualization to {output_file}.")
        
        return extracted_pockets

    def analyze_all_pockets(self, extracted_pockets, search_radius=4.5):
        """
        Analyzes the chemistry for ALL discovered pockets at once.
        Takes the dictionary of pockets and returns a dictionary of results.
        """
        print(f"\nAnalyzing chemistry for {len(extracted_pockets)} pockets...")
        
        # 1. Build the KD-Tree using the ALREADY EXISTING array of atom coordinates
        tree = KDTree(self.atoms_coords)

        # NEW: Calculate the center of the ENTIRE protein (Done ONCE)
        protein_com = np.mean(self.atoms_coords, axis=0)
        
        analysis_results = {}
        
        for pocket_id, voxel_indices in extracted_pockets.items():
            
            pocket_coords = (voxel_indices * self.grid_spacing) + self.min_coords
            neighbor_indices = tree.query_ball_point(pocket_coords, r=search_radius)
            
            unique_atom_indices = set([idx for sublist in neighbor_indices for idx in sublist])
            
            # 1. Real Volume in cubic Angstroms
            volume_A3 = len(voxel_indices) * (self.grid_spacing ** 3)
            # 2. Center of Mass of the pocket (average X, Y, Z)
            pocket_com = np.mean(pocket_coords, axis=0)

            # Calculate "Depth Score" (Distance from pocket center to protein center)
            # np.linalg.norm calculates the standard straight-line distance between two 3D points
            depth_score = np.linalg.norm(protein_com - pocket_com)

            # --- EXPANDED ML CHEMISTRY FEATURES ---
            contact_count = {
                "hydrophobic": 0, "polar": 0, "positive": 0, 
                "negative": 0, "aromatic": 0
            }

            touching_residues = set()
            
            for idx in unique_atom_indices:
                atom_info = self.atom_metadata[idx]
                res_name = atom_info['residue_name']
                res_id = atom_info['residue_id']
                chain_id = atom_info['chain_id']

                # Create a readable label, e.g., "LEU 45 (Chain A)"
                residue_label = f"{res_name} {res_id} (Chain {chain_id})"

                # Only count each residue ONCE for the ML features to avoid skewing data
                if residue_label not in touching_residues:
                    touching_residues.add(residue_label)

                    if res_name in self.HYDROPHOBIC:
                        contact_count["hydrophobic"] += 1
                    if res_name in self.POLAR:
                        contact_count["polar"] += 1
                    if res_name in self.POSITIVE:
                        contact_count["positive"] += 1
                    if res_name in self.NEGATIVE:
                        contact_count["negative"] += 1
                    if res_name in self.AROMATIC:
                        contact_count["aromatic"] += 1
                    
            total_residues = len(touching_residues)

            # Save the ML features
            analysis_results[pocket_id] = {
                "volume_A3": volume_A3,
                "distance_to_core": depth_score,
                "hydrophobic_count": contact_count["hydrophobic"],
                "polar_count": contact_count["polar"],
                "positive_count": contact_count["positive"],
                "negative_count": contact_count["negative"],
                "aromatic_count": contact_count["aromatic"],
                "total_residues_touching": total_residues,
                "residue_list": sorted(list(touching_residues), key=lambda x: int(x.split()[1]))
            }
            
            print(f"Pocket {pocket_id}: {volume_A3:.1f} Å³ | {contact_count['aromatic']} Aromatic residues")
            
        return analysis_results

    def export_report(self, analysis_results, output_file="binding_sites_report.txt"):
        """
        Exports a detailed text report of all detected pockets and their 
        contacting amino acids, satisfying the project's output requirement.
        """
        print(f"\nExporting final report to {output_file}...")
        
        with open(output_file, "w") as f:
            f.write("=========================================\n")
            f.write(f"  LIGAND BINDING SITES REPORT: {self.pdb_file}\n")
            f.write("=========================================\n\n")
            
            for pocket_id, data in analysis_results.items():
                f.write(f"POCKET {pocket_id}\n")
                f.write("-" * 40 + "\n")
                # UPDATED TO PRINT THE NEW ML FEATURES
                f.write(f"Volume:           {data['volume_A3']:.1f} Å³\n")
                f.write(f"Distance to Core: {data['distance_to_core']:.1f} Å\n")
                f.write(f"Chemistry Counts: {data['hydrophobic_count']} Hydrophobic, {data['polar_count']} Polar\n")
                f.write(f"                  {data['positive_count']} Positive, {data['negative_count']} Negative\n")
                f.write(f"                  {data['aromatic_count']} Aromatic\n\n")
                
                residues = data['residue_list']
                f.write(f"Residues Involved ({len(residues)} total):\n")
                
                for i in range(0, len(residues), 4):
                    chunk = residues[i:i+4]
                    f.write("  " + ", ".join(chunk) + "\n")
                    
                f.write("\n") 
                
        print(f"Report successfully saved! Check {output_file} for details.")

if __name__ == "__main__":
    import os
    
    # 1. Define your inputs here!
    work_dir = "/home/diegovicente/pyt"
    protein_name = "1STP"
    input_pdb = os.path.join(work_dir, f"{protein_name}.pdb")
    
    # 2. Initialize and map the protein
    my_protein = ProteinGrid(input_pdb)
    my_protein.scan_pockets()
    
    # 3. Create the PyMOL/Chimera visualization file
    out_pdb = os.path.join(work_dir, f"{protein_name}_pockets.pdb")
    my_pockets_dict = my_protein.cluster_and_export_pockets(output_file=out_pdb, min_size=50)

    # 4. Analyze the chemistry of the valid pockets
    chemistry_results = my_protein.analyze_all_pockets(my_pockets_dict)
    
    # 5. GENERATE THE FINAL REPORT!
    out_report = os.path.join(work_dir, f"{protein_name}_binding_sites.txt")
    my_protein.export_report(chemistry_results, output_file=out_report)