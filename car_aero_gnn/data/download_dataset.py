"""
Dataset Download Script

Supports:
1. DrivAerNet: 4000+ vehicle shapes with CFD results
2. Ahmed Body: Standard benchmark geometry
3. OpenFOAM tutorials: motorBike case
"""

import os
import argparse
import urllib.request
import tarfile
import zipfile
from pathlib import Path
import json


class DatasetDownloader:
    """Download and extract CFD datasets"""

    def __init__(self, root_dir='data/raw'):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url, destination):
        """Download file with progress bar"""
        print(f"Downloading {url}...")

        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\rProgress: {percent}%", end='')

        urllib.request.urlretrieve(url, destination, reporthook=progress_hook)
        print("\nDownload complete!")

    def extract_archive(self, archive_path, extract_dir):
        """Extract tar.gz or zip archive"""
        print(f"Extracting {archive_path}...")

        if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
        elif archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

        print(f"Extracted to {extract_dir}")

    def download_ahmed_body(self):
        """
        Download Ahmed body geometry and reference data

        Ahmed body is a simplified car shape used for aerodynamic testing
        Reference: Ahmed et al. (1984)
        """
        print("=" * 70)
        print("Downloading Ahmed Body Dataset")
        print("=" * 70)

        ahmed_dir = self.root_dir / 'ahmed_body'
        ahmed_dir.mkdir(exist_ok=True)

        # Create geometry file (simplified - actual geometry would come from CAD)
        geometry_info = {
            'name': 'Ahmed Body',
            'length': 1.044,  # meters
            'width': 0.389,
            'height': 0.288,
            'slant_angles': [25, 35],  # degrees
            'reference': 'Ahmed, S. R., Ramm, G., & Faltin, G. (1984)',
            'description': 'Simplified car geometry for aerodynamic validation'
        }

        with open(ahmed_dir / 'geometry_info.json', 'w') as f:
            json.dump(geometry_info, f, indent=2)

        print(f"\nAhmed body info saved to {ahmed_dir / 'geometry_info.json'}")
        print("\nNote: You need to create OpenFOAM mesh using blockMesh and snappyHexMesh")
        print("Tutorial available at: https://www.openfoam.com/documentation/tutorial-guide")

        return ahmed_dir

    def download_drivaernet(self):
        """
        Download DrivAerNet dataset

        DrivAerNet contains 4000+ parametric car shapes with CFD simulations
        Paper: "DrivAerNet: A Parametric Car Dataset for Data-Driven Aerodynamic Design"
        """
        print("=" * 70)
        print("Downloading DrivAerNet Dataset")
        print("=" * 70)

        drivaernet_dir = self.root_dir / 'drivaernet'
        drivaernet_dir.mkdir(exist_ok=True)

        print("\nDrivAerNet dataset is available on Kaggle:")
        print("https://www.kaggle.com/datasets/mohamedelrefaie/drivaernet")
        print("\nTo download:")
        print("1. Install Kaggle API: pip install kaggle")
        print("2. Set up API credentials: ~/.kaggle/kaggle.json")
        print("3. Run: kaggle datasets download -d mohamedelrefaie/drivaernet")
        print(f"4. Extract to: {drivaernet_dir}")

        # Save dataset info
        dataset_info = {
            'name': 'DrivAerNet',
            'url': 'https://github.com/Mohamedelrefaie/DrivAerNet',
            'kaggle': 'https://www.kaggle.com/datasets/mohamedelrefaie/drivaernet',
            'num_samples': 4000,
            'parameters': {
                'length': [3.8, 5.2],  # meters
                'width': [1.6, 2.0],
                'height': [1.3, 1.8],
                'ground_clearance': [0.1, 0.3]
            },
            'cfd_solver': 'OpenFOAM',
            'reynolds_number': 5e6
        }

        with open(drivaernet_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)

        return drivaernet_dir

    def download_motorbike(self):
        """
        Download OpenFOAM motorBike tutorial case

        This is a standard OpenFOAM tutorial for external aerodynamics
        """
        print("=" * 70)
        print("Downloading OpenFOAM motorBike Tutorial")
        print("=" * 70)

        motorbike_dir = self.root_dir / 'motorbike'
        motorbike_dir.mkdir(exist_ok=True)

        print("\nOpenFOAM motorBike tutorial is included in OpenFOAM installation:")
        print("$FOAM_TUTORIALS/incompressible/simpleFoam/motorBike")
        print("\nTo use:")
        print("1. Install OpenFOAM: https://openfoam.org/download/")
        print("2. Copy tutorial: cp -r $FOAM_TUTORIALS/incompressible/simpleFoam/motorBike .")
        print("3. Run: ./Allrun")
        print(f"4. Results will be in time directories")

        tutorial_info = {
            'name': 'motorBike',
            'solver': 'simpleFoam',
            'turbulence_model': 'k-omega SST',
            'description': 'Steady-state flow around motorcycle geometry'
        }

        with open(motorbike_dir / 'tutorial_info.json', 'w') as f:
            json.dump(tutorial_info, f, indent=2)

        return motorbike_dir

    def create_sample_data(self):
        """
        Create small sample dataset for testing
        """
        print("=" * 70)
        print("Creating Sample Test Data")
        print("=" * 70)

        sample_dir = self.root_dir / 'sample'
        sample_dir.mkdir(exist_ok=True)

        # Create minimal sample data structure
        sample_info = {
            'name': 'Sample Test Data',
            'description': 'Minimal dataset for testing pipeline',
            'num_samples': 10,
            'nodes_per_sample': 1000,
            'note': 'Replace with real CFD data for training'
        }

        with open(sample_dir / 'sample_info.json', 'w') as f:
            json.dump(sample_info, f, indent=2)

        print(f"\nSample data directory created at {sample_dir}")
        print("Use this for testing the pipeline before downloading large datasets")

        return sample_dir


def main():
    parser = argparse.ArgumentParser(description='Download CFD datasets')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['ahmed_body', 'drivaernet', 'motorbike', 'sample', 'all'],
        default='sample',
        help='Dataset to download'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/raw',
        help='Output directory for downloaded data'
    )

    args = parser.parse_args()

    downloader = DatasetDownloader(args.output_dir)

    if args.dataset == 'ahmed_body' or args.dataset == 'all':
        downloader.download_ahmed_body()

    if args.dataset == 'drivaernet' or args.dataset == 'all':
        downloader.download_drivaernet()

    if args.dataset == 'motorbike' or args.dataset == 'all':
        downloader.download_motorbike()

    if args.dataset == 'sample':
        downloader.create_sample_data()

    print("\n" + "=" * 70)
    print("Download process complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Set up CFD simulations (OpenFOAM, STAR-CCM+, etc.)")
    print("2. Run preprocess.py to convert CFD results to graph format")
    print("3. Start training with main.py")


if __name__ == '__main__':
    main()
