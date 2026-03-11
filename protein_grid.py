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
    STANDARD_AAS = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
                    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

    def __init__(self, pdb_file, grid_spacing=1.0, padding=3.0):
        """
        Initialize the class with the PDB file, the size of each voxel (grid_spacing),
        and the extra margin around the protein (padding).
        """

        self.pdb_file = pdb_file
        self.protein_name = self.pdb_file.split("/")[-1].split(".")[0]
        self.grid_spacing = grid_spacing
        self.padding = padding
        self.atoms_coords = []
        self.atom_metadata = []
        self.ligand_coords = []
        self.grid = None
        self.grid_shape = None
        
        # Execute the initial setup methods
        self.extract_coordinates()
        self.calculate_bounding_box()
        self.map_to_grid()

    def extract_coordinates(self):
        """
        Use Biopython to extract the X, Y, Z coordinates of all atoms in the protein.
        """

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", self.pdb_file)
        print(f"Extracting coordinates from {self.protein_name}...")
        
        # Iterate through the model, chains, residues, and atoms of the protein structure
        for model in structure:
            for chain in model:
                for residue in chain:

                    res_type = residue.get_id()[0]

                    # Skip water molecules since they are not part of the protein or ligand
                    if res_type == 'W':
                        continue

                    # If it's a Ligand/Heteroatom, save its coordinates for the ML label
                    elif res_type.startswith('H_'):
                        for atom in residue:
                            self.ligand_coords.append(atom.coord)
                        continue # Move to the next residue, don't add ligand residues to the protein grid

                    # Otherwise, it is a standard amino acid (res_type == ' ')
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
        print(f"Extracted {len(self.atoms_coords)} protein atoms from {self.protein_name}.")

        if self.ligand_coords:
            self.ligand_coords = np.array(self.ligand_coords)
            print(f"Extracted {len(self.ligand_coords)} ligand atoms from {self.protein_name}.")

    def calculate_bounding_box(self):
        """
        Calculate the boundaries of the 3D bounding box that encloses the protein.
        """

        # Find the minimum and maximum X, Y, Z values among all atoms
        self.min_coords = np.min(self.atoms_coords, axis=0) - self.padding
        max_coords = np.max(self.atoms_coords, axis=0) + self.padding
        
        # Calculate the overall dimensions of the bounding box
        box_dimensions = max_coords - self.min_coords
        
        # Determine how many 3D pixels (voxels) fit within the bounding box
        self.grid_shape = np.ceil(box_dimensions / self.grid_spacing).astype(int)
        print(f"Grid shape (X, Y, Z) of {self.protein_name}: {self.grid_shape}")
        
        # Create an empty (all zeros) 3D grid array for storing the protein's presence (1s) and empty space (0s)
        self.grid = np.zeros(self.grid_shape, dtype=int)
    
    def map_to_grid(self):
        """
        Maps the continuous 3D coordinates into the discrete 3D grid.
        Assigns a value of 1 to voxels containing at least one atom.
        """

        print(f"Mapping atoms of {self.protein_name} to 3D grid...")
        
        for coord in self.atoms_coords:
            # Shift the coordinates so the minimum of the bounding box is at (0,0,0)
            shifted_coord = coord - self.min_coords
            
            # Divide by grid_spacing to find the corresponding matrix index
            grid_index = (shifted_coord / self.grid_spacing).astype(int)
            
            # Extract X, Y, Z matrix indices
            ix, iy, iz = grid_index
            
            # Mark the specific voxel as occupied
            self.grid[ix, iy, iz] = 1
            
        occupied_voxels = np.sum(self.grid)
        print(f"Success: {len(self.atoms_coords)} atoms mapped into {occupied_voxels} out of {self.grid.size} voxels.")

    def scan_pockets(self):
        """
        Scans the 3D grid along the X, Y, and Z axes to find empty spaces (0s)
        that are trapped between protein atoms (1s) on at least two sides, indicating potential pockets.
        """

        print(f"\nStarting LIGSITE 3-axis scan for {self.protein_name}...")
        
        # 1. Sweep along the X-axis (Left and Right)
        # Check if there is a '1' anywhere to the left, and anywhere to the right
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
        
        # 4. Count the number of trapping axes (protein-solvent-protein (PSP) events)
        psp_count = bounded_x.astype(int) + bounded_y.astype(int) + bounded_z.astype(int)
        
        # 5. Identify the true surface pockets (must be empty and trapped in at least 2 directions)
        is_empty = (self.grid == 0)
        self.pocket_grid = is_empty & (psp_count >= 2)
        
        pocket_volume = np.sum(self.pocket_grid)
        print(f"Scan complete! Found {pocket_volume} pocket voxels in {self.protein_name}.")

    def cluster_and_export_pockets(self, output_file=None, min_size=50):
        """
        Groups adjacent pocket voxels into distinct pockets, filters out tiny ones,
        and exports the valid pockets' coordinates. If output_file is provided, it also
        saves them as dummy atoms in a PDB file for structural visualization.
        """

        print(f"\nClustering pockets and filtering out those smaller than {min_size} voxels...")

        # Group connected voxels with the numpy label function (clusters together adjacent 1s in 'pocket_grid' into distinct pockets).
        # 'labels' is a new grid where Pocket 1 is filled with 1s, Pocket 2 with 2s, etc. 'num_features' is the total number of pockets.
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
            real_coords = (voxel_indices * self.grid_spacing) + self.min_coords
            extracted_pockets[valid_pockets] = real_coords

            print(f"Pocket {valid_pockets}: {len(voxel_indices)} voxels")
        
        # Only write the PDB file if an output_file string was actually provided
        if output_file:

            with open(output_file, "w") as f:
                atom_serial = 1

                for valid_id, real_coords in extracted_pockets.items():

                    for coord in real_coords:

                        x, y, z = coord
                        # write a HETATM line for each pocket voxel, using the valid_id to differentiate pockets and the real_coords for the position
                        f.write(
                            f"HETATM{atom_serial:5d}  POC STP A{valid_id:4d}    "
                            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n"
                        )

                        atom_serial += 1

            print(f"Saved {self.protein_name}'s pockets to {output_file} for visualization in PyMOL or Chimera")
        
        return extracted_pockets

    def analyze_all_pockets(self, extracted_pockets, search_radius=4.5):
        """
        Analyzes the chemistry for ALL discovered pockets at once.
        Takes the dictionary of pockets and returns a dictionary of results.
        """
        
        print(f"\nAnalyzing chemistry for {len(extracted_pockets)} pockets...")
        tree = KDTree(self.atoms_coords)

        # Calculate the center of the protein once
        protein_com = np.mean(self.atoms_coords, axis=0)

        # Build a tree for the ligand if one exists in the PDB
        if len(self.ligand_coords) > 0:
            ligand_tree = KDTree(self.ligand_coords)
        else:
            ligand_tree = None
        
        analysis_results = {}
        
        for pocket_id, pocket_coords in extracted_pockets.items():
            
            # Calculate the ML Label (is this pocket the true binding site?)
            is_binding_site = 0

            if ligand_tree is not None:
                # Check if any pocket voxel is within 3.0 Angstroms of any ligand atom
                ligand_contacts = ligand_tree.query_ball_point(pocket_coords, r=3.0)
                if any(ligand_contacts): 
                    is_binding_site = 1

            # Find all protein atoms within the search_radius of any pocket voxel
            neighbor_indices = tree.query_ball_point(pocket_coords, r=search_radius)
            unique_atom_indices = set([idx for sublist in neighbor_indices for idx in sublist])
            
            # --- CALCULATE ML FEATURES ---
            # Real volume in cubic Angstroms
            volume_A3 = len(pocket_coords) * (self.grid_spacing ** 3)

            # Depth of the pocket (Distance from pocket center to protein center)
            pocket_com = np.mean(pocket_coords, axis=0)
            depth_score = np.linalg.norm(protein_com - pocket_com)

            # Dictionary to count the types of amino acids in contact with the pocket
            contact_count = {
                "hydrophobic": 0, "polar": 0, "positive": 0, 
                "negative": 0, "aromatic": 0
            }

            # Dictionary to count the proportions of each standard amino acid type in contact with the pocket
            aa_counts = {aa: 0 for aa in self.STANDARD_AAS}

            # Set to keep track of unique residues touching the pocket
            touching_residues = set()
            
            # Loop through each unique atom index that is in contact with the pocket
            for idx in unique_atom_indices:
                atom_info = self.atom_metadata[idx]

                res_name = atom_info['residue_name']
                res_id = atom_info['residue_id']
                chain_id = atom_info['chain_id']

                # Create a readable label for each residue"
                residue_label = f"{res_name} {res_id} (Chain {chain_id})"

                # Only count each residue once for the ML features to avoid skewing data
                if residue_label not in touching_residues:
                    touching_residues.add(residue_label)

                    if res_name in aa_counts:
                        aa_counts[res_name] += 1

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

            # Calculate proportions (avoiding division by zero)
            if total_residues > 0:
                prop_hydro = contact_count["hydrophobic"] / total_residues
                prop_polar = contact_count["polar"] / total_residues
                prop_pos = contact_count["positive"] / total_residues
                prop_neg = contact_count["negative"] / total_residues
                prop_arom = contact_count["aromatic"] / total_residues
            else:
                prop_hydro = prop_polar = prop_pos = prop_neg = prop_arom = 0.0

            # Calculate specific amino acid proportions
            aa_proportions = {}
            for aa in self.STANDARD_AAS:
                if total_residues > 0:
                    aa_proportions[aa] = aa_counts[aa] / total_residues
                else:
                    aa_proportions[aa] = 0.0

            # Save the ML features
            analysis_results[pocket_id] = {
                "volume_A3": volume_A3,
                "distance_to_core": depth_score,
                "hydrophobic_count": contact_count["hydrophobic"],
                "polar_count": contact_count["polar"],
                "positive_count": contact_count["positive"],
                "negative_count": contact_count["negative"],
                "aromatic_count": contact_count["aromatic"],
                "hydrophobic_prop": prop_hydro,  
                "polar_prop": prop_polar,        
                "positive_prop": prop_pos,       
                "negative_prop": prop_neg,       
                "aromatic_prop": prop_arom,      
                "total_residues_touching": total_residues,
                "residue_list": sorted(list(touching_residues), key=lambda x: int(x.split()[1])),
                **aa_proportions,
                "is_binding_site": is_binding_site
            }
            
            print(f"Pocket {pocket_id} analyzed: Volume={volume_A3:.1f} Å³, Depth={depth_score:.1f} Å, Contacts={total_residues}, Binding Site={is_binding_site}")
            
        return analysis_results

    def export_report(self, analysis_results, output_file="binding_sites_report.txt"):
        """
        Exports a detailed text report of all detected pockets and their 
        contacting amino acids, along with the ML features and whether they are true binding sites.
        """

        print(f"\nExporting final report of {len(analysis_results)} pockets from {self.pdb_file} to {output_file}...")
        
        with open(output_file, "w") as f:
            f.write("=========================================\n")
            f.write(f"  LIGAND BINDING SITES REPORT: {self.pdb_file}\n")
            f.write("=========================================\n\n")
            
            for pocket_id, data in analysis_results.items():
                f.write(f"POCKET {pocket_id}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Volume:           {data['volume_A3']:.1f} Å³\n")
                f.write(f"Distance to Core: {data['distance_to_core']:.1f} Å\n")
                f.write(f"Chemistry Counts: {data['hydrophobic_count']} Hydrophobic, {data['polar_count']} Polar\n")
                f.write(f"                  {data['positive_count']} Positive, {data['negative_count']} Negative\n")
                f.write(f"                  {data['aromatic_count']} Aromatic\n\n")
                
                residues = data['residue_list']
                f.write(f"Residues Involved ({len(residues)} total):\n")
                f.write(f"It is a true binding site: {'YES' if data['is_binding_site'] == 1 else 'NO'}\n")
                
                for i in range(0, len(residues), 4):
                    chunk = residues[i:i+4]
                    f.write("  " + ", ".join(chunk) + "\n")
                    
                f.write("\n\n") 
                
        print(f"Report successfully saved! Check {output_file} for details.")

if __name__ == "__main__":
    import os
    
    # Define your inputs here
    work_dir = "/Users/pauvillen14/Desktop/BIOINFO/SBI/PROJECT/pred_binding-site"
    protein_name = "1STP"
    input_pdb = os.path.join(work_dir, f"{protein_name}.pdb")
    
    # Initialize and map the protein
    my_protein = ProteinGrid(input_pdb)
    my_protein.scan_pockets()
    
    # Create the PyMOL/Chimera visualization file
    out_pdb = os.path.join(work_dir, f"{protein_name}_pockets.pdb")
    my_pockets_dict = my_protein.cluster_and_export_pockets(output_file=out_pdb, min_size=50)

    # Analyze the chemistry of the valid pockets
    chemistry_results = my_protein.analyze_all_pockets(my_pockets_dict)
    
    # Generate the final report
    out_report = os.path.join(work_dir, f"{protein_name}_binding_sites.txt")
    my_protein.export_report(chemistry_results, output_file=out_report)