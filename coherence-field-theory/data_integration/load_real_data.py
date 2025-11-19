"""
Load real observational data for coherence field theory tests.

This module loads actual SPARC, Pantheon, and cluster data from ../data/
"""

import numpy as np
import pandas as pd
import os
import glob


class RealDataLoader:
    """
    Load real observational data.
    """
    
    def __init__(self, base_data_dir=None):
        """
        Parameters:
        -----------
        base_data_dir : str
            Path to data directory (relative to coherence-field-theory/)
        """
        if base_data_dir is None:
            # Try to find data directory automatically
            possible_paths = [
                '../../data',  # From coherence-field-theory/data_integration/
                '../data',     # From coherence-field-theory/
                'data',        # From repo root
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    base_data_dir = path
                    break
            if base_data_dir is None:
                base_data_dir = '../../data'
        
        self.base_data_dir = base_data_dir
        print(f"Using data directory: {os.path.abspath(self.base_data_dir)}")
        
    def load_rotmod_galaxy(self, galaxy_name):
        """
        Load rotation curve from Rotmod_LTG directory.
        
        Parameters:
        -----------
        galaxy_name : str
            Galaxy name (e.g., 'DDO154', 'NGC2403')
            
        Returns:
        --------
        data : dict
            Dictionary with:
            - name: galaxy name
            - r: radii (kpc)
            - v_obs: observed velocities (km/s)
            - v_err: velocity uncertainties
            - v_disk: disk component
            - v_gas: gas component
            - v_bulge: bulge component (if present)
            - distance: distance in Mpc
        """
        rotmod_dir = os.path.join(self.base_data_dir, 'Rotmod_LTG')
        
        # Try different filename formats
        possible_files = [
            os.path.join(rotmod_dir, f'{galaxy_name}_rotmod.dat'),
            os.path.join(rotmod_dir, f'{galaxy_name.upper()}_rotmod.dat'),
            os.path.join(rotmod_dir, f'{galaxy_name.lower()}_rotmod.dat'),
        ]
        
        filepath = None
        for f in possible_files:
            if os.path.exists(f):
                filepath = f
                break
        
        if filepath is None:
            raise FileNotFoundError(f"Could not find rotation curve for {galaxy_name}")
        
        print(f"Loading: {filepath}")
        
        # Read the file
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Extract distance from first comment line
        distance = None
        for line in lines:
            if line.startswith('# Distance'):
                # Format: "# Distance = X.XX Mpc"
                distance = float(line.split('=')[1].split('Mpc')[0].strip())
                break
        
        # Load data (skip comment lines starting with #)
        df = pd.read_csv(filepath, sep=r'\s+', comment='#', 
                        names=['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul'])
        
        data = {
            'name': galaxy_name,
            'r': df['Rad'].values,
            'v_obs': df['Vobs'].values,
            'v_err': df['errV'].values,
            'v_disk': df['Vdisk'].values,
            'v_gas': df['Vgas'].values,
            'v_bulge': df['Vbul'].values if 'Vbul' in df else np.zeros_like(df['Rad'].values),
            'distance_Mpc': distance
        }
        
        print(f"  {len(df)} data points")
        print(f"  Distance: {distance} Mpc")
        print(f"  Radius range: {data['r'][0]:.2f} - {data['r'][-1]:.2f} kpc")
        print(f"  Velocity range: {data['v_obs'].min():.1f} - {data['v_obs'].max():.1f} km/s")
        
        return data
    
    def list_available_galaxies(self):
        """
        List all available galaxies in Rotmod_LTG.
        
        Returns:
        --------
        galaxies : list
            List of galaxy names
        """
        rotmod_dir = os.path.join(self.base_data_dir, 'Rotmod_LTG')
        
        if not os.path.exists(rotmod_dir):
            print(f"Rotmod_LTG directory not found: {rotmod_dir}")
            return []
        
        files = glob.glob(os.path.join(rotmod_dir, '*_rotmod.dat'))
        galaxies = [os.path.basename(f).replace('_rotmod.dat', '') for f in files]
        
        print(f"\nFound {len(galaxies)} galaxies in Rotmod_LTG:")
        for i, gal in enumerate(sorted(galaxies)[:10]):
            print(f"  {gal}")
        if len(galaxies) > 10:
            print(f"  ... and {len(galaxies)-10} more")
        
        return sorted(galaxies)
    
    def load_pantheon_sample(self, max_z=2.0, max_SNe=None):
        """
        Load Pantheon+ supernova sample.
        
        Parameters:
        -----------
        max_z : float
            Maximum redshift
        max_SNe : int
            Maximum number of SNe (None = all)
            
        Returns:
        --------
        data : dict
            Dictionary with:
            - z: redshifts
            - mu: distance moduli
            - mu_err: uncertainties
        """
        pantheon_file = os.path.join(self.base_data_dir, 'pantheon', 'Pantheon+SH0ES.dat')
        
        if not os.path.exists(pantheon_file):
            raise FileNotFoundError(f"Pantheon data not found: {pantheon_file}")
        
        print(f"\nLoading Pantheon+ data from: {pantheon_file}")
        
        # Load data (space-separated, has header)
        df = pd.read_csv(pantheon_file, sep=r'\s+')
        
        # Use CMB frame redshift and SH0ES distance modulus
        z = df['zCMB'].values
        mu = df['MU_SH0ES'].values
        mu_err = df['MU_SH0ES_ERR_DIAG'].values
        
        # Filter by redshift
        mask = (z > 0) & (z <= max_z)
        z = z[mask]
        mu = mu[mask]
        mu_err = mu_err[mask]
        
        # Limit sample size if requested
        if max_SNe is not None and len(z) > max_SNe:
            indices = np.linspace(0, len(z)-1, max_SNe).astype(int)
            z = z[indices]
            mu = mu[indices]
            mu_err = mu_err[indices]
        
        print(f"  Loaded {len(z)} SNe")
        print(f"  Redshift range: {z.min():.4f} - {z.max():.4f}")
        print(f"  Distance modulus range: {mu.min():.2f} - {mu.max():.2f}")
        
        return {
            'z': z,
            'mu': mu,
            'mu_err': mu_err
        }
    
    def load_cluster_profiles(self, cluster_name):
        """
        Load cluster mass/lensing profiles.
        
        Parameters:
        -----------
        cluster_name : str
            Cluster name (e.g., 'ABELL_1689')
            
        Returns:
        --------
        data : dict
            Dictionary with available profiles
        """
        cluster_dir = os.path.join(self.base_data_dir, 'clusters', cluster_name)
        
        if not os.path.exists(cluster_dir):
            raise FileNotFoundError(f"Cluster directory not found: {cluster_dir}")
        
        print(f"\nLoading cluster: {cluster_name}")
        
        data = {'name': cluster_name}
        
        # Look for different profile files
        profile_files = {
            'stars': 'stars_profile.csv',
            'gas': 'gas_profile.csv',
            'clump': 'clump_profile.csv',
            'temp': 'temp_profile.csv'
        }
        
        for profile_type, filename in profile_files.items():
            filepath = os.path.join(cluster_dir, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                data[profile_type] = df
                print(f"  Loaded {profile_type}: {len(df)} points")
        
        return data
    
    def list_available_clusters(self):
        """
        List available clusters.
        
        Returns:
        --------
        clusters : list
            List of cluster names
        """
        clusters_dir = os.path.join(self.base_data_dir, 'clusters')
        
        if not os.path.exists(clusters_dir):
            print(f"Clusters directory not found: {clusters_dir}")
            return []
        
        # Look for subdirectories that look like cluster names
        clusters = []
        for item in os.listdir(clusters_dir):
            item_path = os.path.join(clusters_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                clusters.append(item)
        
        print(f"\nFound {len(clusters)} clusters:")
        for cluster in sorted(clusters):
            print(f"  {cluster}")
        
        return sorted(clusters)


def demo_load_all_data():
    """Demonstrate loading all data types."""
    print("=" * 80)
    print("REAL DATA LOADING DEMO")
    print("=" * 80)
    
    loader = RealDataLoader()
    
    # 1. List available galaxies
    print("\n" + "=" * 80)
    print("1. GALAXIES (Rotmod_LTG)")
    print("=" * 80)
    galaxies = loader.list_available_galaxies()
    
    # 2. Load example galaxy
    if galaxies:
        print("\n" + "=" * 80)
        print("2. LOAD EXAMPLE GALAXY")
        print("=" * 80)
        example_gal = galaxies[10] if len(galaxies) > 10 else galaxies[0]
        gal_data = loader.load_rotmod_galaxy(example_gal)
    
    # 3. Load Pantheon sample
    print("\n" + "=" * 80)
    print("3. PANTHEON SUPERNOVAE")
    print("=" * 80)
    try:
        pantheon_data = loader.load_pantheon_sample(max_z=1.5, max_SNe=100)
    except Exception as e:
        print(f"Could not load Pantheon: {e}")
    
    # 4. List clusters
    print("\n" + "=" * 80)
    print("4. GALAXY CLUSTERS")
    print("=" * 80)
    clusters = loader.list_available_clusters()
    
    # 5. Load example cluster
    if clusters:
        print("\n" + "=" * 80)
        print("5. LOAD EXAMPLE CLUSTER")
        print("=" * 80)
        example_cluster = clusters[0]
        cluster_data = loader.load_cluster_profiles(example_cluster)
    
    print("\n" + "=" * 80)
    print("DATA LOADING DEMO COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    demo_load_all_data()

