"""
Data loading utilities for Gaia, SPARC, and cluster data.

Provides unified interface to existing data in ../data/ directory.
"""

import numpy as np
import pandas as pd
import os
import glob


class DataLoader:
    """
    Load observational data for coherence field theory tests.
    """
    
    def __init__(self, base_data_dir='../../data'):
        """
        Parameters:
        -----------
        base_data_dir : str
            Path to data directory (relative to coherence-field-theory/)
        """
        self.base_data_dir = base_data_dir
        
    def load_sparc_catalog(self):
        """
        Load SPARC galaxy catalog.
        
        Returns:
        --------
        catalog : DataFrame
            SPARC catalog with galaxy properties
        """
        sparc_file = os.path.join(self.base_data_dir, 'sparc', 'sparc.csv')
        
        if not os.path.exists(sparc_file):
            print(f"Warning: SPARC catalog not found at {sparc_file}")
            return None
        
        try:
            catalog = pd.read_csv(sparc_file)
            print(f"Loaded SPARC catalog: {len(catalog)} galaxies")
            return catalog
        except Exception as e:
            print(f"Error loading SPARC catalog: {e}")
            return None
    
    def load_sparc_rotation_curve(self, galaxy_name):
        """
        Load rotation curve for specific SPARC galaxy.
        
        Parameters:
        -----------
        galaxy_name : str
            Galaxy identifier
            
        Returns:
        --------
        data : DataFrame
            Rotation curve data
        """
        # Try Rotmod_LTG directory first (has individual galaxy files)
        rotmod_dir = os.path.join(self.base_data_dir, 'Rotmod_LTG')
        
        if os.path.exists(rotmod_dir):
            # Look for matching file
            pattern = os.path.join(rotmod_dir, f"*{galaxy_name}*.dat")
            files = glob.glob(pattern)
            
            if files:
                try:
                    # Load .dat file (format may vary)
                    data = pd.read_csv(files[0], delim_whitespace=True, 
                                     comment='#', header=None)
                    print(f"Loaded rotation curve for {galaxy_name}")
                    return data
                except Exception as e:
                    print(f"Error loading {files[0]}: {e}")
        
        print(f"Rotation curve for {galaxy_name} not found")
        return None
    
    def load_gaia_wide_binaries(self):
        """
        Load Gaia wide binary data for testing gravity at low accelerations.
        
        Returns:
        --------
        data : DataFrame
            Wide binary data
        """
        gaia_dir = os.path.join(self.base_data_dir, 'gaia')
        
        if not os.path.exists(gaia_dir):
            print(f"Gaia directory not found: {gaia_dir}")
            return None
        
        # Look for relevant files
        csv_files = glob.glob(os.path.join(gaia_dir, '*.csv'))
        
        if csv_files:
            print(f"Found {len(csv_files)} Gaia CSV files")
            # Load first one as example
            try:
                data = pd.read_csv(csv_files[0])
                print(f"Loaded Gaia data: {csv_files[0]}")
                return data
            except Exception as e:
                print(f"Error loading Gaia data: {e}")
        
        return None
    
    def load_cluster_data(self):
        """
        Load galaxy cluster data (Abell clusters, etc.).
        
        Returns:
        --------
        data : DataFrame
            Cluster data
        """
        cluster_dir = os.path.join(self.base_data_dir, 'clusters')
        
        if not os.path.exists(cluster_dir):
            print(f"Cluster directory not found: {cluster_dir}")
            return None
        
        csv_files = glob.glob(os.path.join(cluster_dir, '*.csv'))
        
        if csv_files:
            print(f"Found {len(csv_files)} cluster CSV files")
            # Load first one as example
            try:
                data = pd.read_csv(csv_files[0])
                print(f"Loaded cluster data: {csv_files[0]}")
                return data
            except Exception as e:
                print(f"Error loading cluster data: {e}")
        
        return None
    
    def list_available_data(self):
        """Print summary of available data."""
        print("=" * 70)
        print("Available Data Summary")
        print("=" * 70)
        
        # Check each directory
        dirs_to_check = ['sparc', 'gaia', 'clusters', 'Rotmod_LTG', 'pantheon']
        
        for dirname in dirs_to_check:
            dirpath = os.path.join(self.base_data_dir, dirname)
            if os.path.exists(dirpath):
                files = os.listdir(dirpath)
                print(f"\n{dirname}/:")
                print(f"  {len(files)} files")
                
                # Count by extension
                extensions = {}
                for f in files:
                    ext = os.path.splitext(f)[1]
                    extensions[ext] = extensions.get(ext, 0) + 1
                
                for ext, count in sorted(extensions.items()):
                    print(f"    {count} {ext} files")
            else:
                print(f"\n{dirname}/: NOT FOUND")
        
        print("\n" + "=" * 70)


def main():
    """Test data loading."""
    print("=" * 70)
    print("Data Integration - Testing Data Loading")
    print("=" * 70)
    
    loader = DataLoader()
    
    # List available data
    loader.list_available_data()
    
    # Try loading each dataset
    print("\n" + "=" * 70)
    print("Attempting to load datasets...")
    print("=" * 70)
    
    print("\n1. SPARC catalog:")
    sparc_cat = loader.load_sparc_catalog()
    if sparc_cat is not None:
        print(f"   Columns: {list(sparc_cat.columns)}")
    
    print("\n2. Gaia data:")
    gaia_data = loader.load_gaia_wide_binaries()
    if gaia_data is not None:
        print(f"   Shape: {gaia_data.shape}")
    
    print("\n3. Cluster data:")
    cluster_data = loader.load_cluster_data()
    if cluster_data is not None:
        print(f"   Shape: {cluster_data.shape}")
    
    print("\n" + "=" * 70)
    print("Data loading test complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

