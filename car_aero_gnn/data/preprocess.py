"""
CFD Data Preprocessing Pipeline

Converts CFD mesh and field data into PyTorch Geometric graph format

Input: VTK, OpenFOAM, or other mesh formats
Output: PyG Data objects with node/edge features
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph, knn_graph
from pathlib import Path
from tqdm import tqdm
import json


try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("Warning: PyVista not installed. Limited mesh format support.")


class MeshToGraphConverter:
    """
    Convert CFD mesh to graph representation

    Graph structure:
    - Nodes: Mesh points with features (position, velocity, pressure, etc.)
    - Edges: Connectivity (mesh topology + spatial proximity)
    """

    def __init__(self, config):
        self.config = config
        self.radius = config.get('edge_radius', 0.1)  # For radius_graph
        self.max_neighbors = config.get('max_neighbors', 10)  # For knn_graph
        self.edge_mode = config.get('edge_mode', 'radius')  # 'radius' or 'knn'

    def load_vtk_mesh(self, vtk_path):
        """
        Load mesh from VTK file

        Args:
            vtk_path: Path to VTK file

        Returns:
            mesh: PyVista mesh object
        """
        if not HAS_PYVISTA:
            raise ImportError("PyVista required for VTK loading. pip install pyvista")

        mesh = pv.read(vtk_path)
        return mesh

    def load_openfoam_case(self, case_dir, time_step='latest'):
        """
        Load OpenFOAM case data

        Args:
            case_dir: OpenFOAM case directory
            time_step: Time step to load ('latest' or specific time)

        Returns:
            mesh: PyVista mesh with field data
        """
        if not HAS_PYVISTA:
            raise ImportError("PyVista required. pip install pyvista")

        from pyvista import OpenFOAMReader

        reader = OpenFOAMReader(case_dir)
        mesh = reader.read()

        # Get latest time step if requested
        if time_step == 'latest':
            times = reader.time_values
            if len(times) > 0:
                reader.set_active_time_value(times[-1])
                mesh = reader.read()

        return mesh

    def compute_wall_distance(self, positions, node_types):
        """
        Compute distance to nearest wall for each node

        Args:
            positions: [N, 3] node coordinates
            node_types: [N] node type labels (0=interior, 1=wall, 2=boundary)

        Returns:
            wall_dist: [N, 1] distance to wall
        """
        wall_mask = (node_types == 1)
        wall_positions = positions[wall_mask]

        if len(wall_positions) == 0:
            # No wall nodes, return zeros
            return torch.zeros(len(positions), 1)

        # Compute pairwise distances
        # For large meshes, use chunked computation
        chunk_size = 10000
        wall_dist = torch.zeros(len(positions), 1)

        for i in range(0, len(positions), chunk_size):
            end_i = min(i + chunk_size, len(positions))
            pos_chunk = positions[i:end_i]

            # Compute distance to all wall points
            dist = torch.cdist(pos_chunk.unsqueeze(0), wall_positions.unsqueeze(0))
            wall_dist[i:end_i] = dist.min(dim=-1)[0].squeeze(0).unsqueeze(-1)

        return wall_dist

    def create_edge_index(self, positions, mesh=None):
        """
        Create graph edge connectivity

        Two strategies:
        1. Radius-based: Connect nodes within distance r
        2. KNN-based: Connect k nearest neighbors

        Args:
            positions: [N, 3] node coordinates
            mesh: Optional mesh object for topology info

        Returns:
            edge_index: [2, E] edge connectivity
        """
        if self.edge_mode == 'radius':
            edge_index = radius_graph(
                positions,
                r=self.radius,
                loop=False,
                max_num_neighbors=self.max_neighbors * 2
            )
        elif self.edge_mode == 'knn':
            edge_index = knn_graph(
                positions,
                k=self.max_neighbors,
                loop=False
            )
        else:
            raise ValueError(f"Unknown edge_mode: {self.edge_mode}")

        # Optionally add mesh topology edges
        if mesh is not None and hasattr(mesh, 'faces'):
            mesh_edges = self.extract_mesh_edges(mesh)
            edge_index = torch.cat([edge_index, mesh_edges], dim=1)

            # Remove duplicates
            edge_index = torch.unique(edge_index, dim=1)

        return edge_index

    def extract_mesh_edges(self, mesh):
        """
        Extract edges from mesh face connectivity

        Args:
            mesh: PyVista mesh

        Returns:
            edges: [2, E] edge connectivity
        """
        faces = mesh.faces.reshape(-1, 4)[:, 1:]  # Remove face size prefix
        edges = []

        for face in faces:
            # Add edges between consecutive vertices in face
            for i in range(len(face)):
                j = (i + 1) % len(face)
                edges.append([face[i], face[j]])
                edges.append([face[j], face[i]])  # Bidirectional

        edges = torch.tensor(edges, dtype=torch.long).t()
        return edges

    def create_edge_features(self, positions, edge_index):
        """
        Create edge features from node positions

        Features:
        - Relative position: dx, dy, dz
        - Euclidean distance: ||r||

        Args:
            positions: [N, 3] node positions
            edge_index: [2, E] edge connectivity

        Returns:
            edge_attr: [E, 4] edge features
        """
        row, col = edge_index

        # Relative position
        relative_pos = positions[col] - positions[row]

        # Euclidean distance
        distance = torch.norm(relative_pos, dim=1, keepdim=True)

        edge_attr = torch.cat([relative_pos, distance], dim=-1)

        return edge_attr

    def mesh_to_graph(
        self,
        mesh,
        velocity_field='U',
        pressure_field='p',
        normalize=True
    ):
        """
        Convert mesh with field data to graph

        Args:
            mesh: PyVista mesh with field data
            velocity_field: Name of velocity field
            pressure_field: Name of pressure field
            normalize: Whether to normalize features

        Returns:
            data: PyG Data object
        """
        # Extract positions
        positions = torch.tensor(mesh.points, dtype=torch.float32)

        # Extract velocity field
        if velocity_field in mesh.point_data:
            velocity = torch.tensor(mesh.point_data[velocity_field], dtype=torch.float32)
        else:
            print(f"Warning: Velocity field '{velocity_field}' not found. Using zeros.")
            velocity = torch.zeros(len(positions), 3)

        # Extract pressure field
        if pressure_field in mesh.point_data:
            pressure = torch.tensor(mesh.point_data[pressure_field], dtype=torch.float32)
            if pressure.dim() == 1:
                pressure = pressure.unsqueeze(-1)
        else:
            print(f"Warning: Pressure field '{pressure_field}' not found. Using zeros.")
            pressure = torch.zeros(len(positions), 1)

        # Determine node types (simplified - actual implementation needs boundary info)
        # 0: interior, 1: wall, 2: other boundary
        node_types = self.determine_node_types(mesh, positions)

        # Compute wall distance
        wall_dist = self.compute_wall_distance(positions, node_types)

        # One-hot encode node types
        node_type_onehot = F.one_hot(node_types, num_classes=3).float()

        # Normalize features if requested
        if normalize:
            positions_normalized = self.normalize_positions(positions)
            velocity_normalized = self.normalize_velocity(velocity)
            pressure_normalized = self.normalize_pressure(pressure)
        else:
            positions_normalized = positions
            velocity_normalized = velocity
            pressure_normalized = pressure

        # Construct node features
        # [pos (3), vel (3), pressure (1), wall_dist (1), type (3)] = 11 dims
        x = torch.cat([
            positions_normalized,
            velocity_normalized,
            pressure_normalized,
            wall_dist,
            node_type_onehot
        ], dim=-1)

        # Create edges
        edge_index = self.create_edge_index(positions, mesh)

        # Create edge features
        edge_attr = self.create_edge_features(positions, edge_index)

        # Create target (for supervised learning)
        y = torch.cat([velocity, pressure], dim=-1)

        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=positions,
            y=y
        )

        return data

    def determine_node_types(self, mesh, positions):
        """
        Determine node types (interior, wall, boundary)

        This is a simplified version. Actual implementation should use
        boundary patch information from CFD solver.

        Args:
            mesh: PyVista mesh
            positions: Node positions

        Returns:
            node_types: [N] tensor of node type labels
        """
        # Simplified: classify by z-coordinate (ground as wall)
        node_types = torch.zeros(len(positions), dtype=torch.long)

        # Mark ground (z ≈ 0) as wall
        ground_threshold = positions[:, 2].min() + 0.01
        wall_mask = positions[:, 2] < ground_threshold
        node_types[wall_mask] = 1

        # Mark far boundaries as boundary type
        # (Simplified - should use actual boundary patch info)
        bounds = positions.numpy()
        x_max, y_max, z_max = bounds.max(axis=0)
        boundary_threshold = 0.95

        boundary_mask = (
            (positions[:, 0] > x_max * boundary_threshold) |
            (positions[:, 1].abs() > y_max * boundary_threshold) |
            (positions[:, 2] > z_max * boundary_threshold)
        )
        node_types[boundary_mask] = 2

        return node_types

    def normalize_positions(self, positions):
        """Normalize positions to [-1, 1]"""
        min_pos = positions.min(dim=0)[0]
        max_pos = positions.max(dim=0)[0]
        return 2 * (positions - min_pos) / (max_pos - min_pos + 1e-8) - 1

    def normalize_velocity(self, velocity):
        """Normalize velocity by maximum magnitude"""
        vel_mag = torch.norm(velocity, dim=1, keepdim=True)
        max_vel = vel_mag.max()
        return velocity / (max_vel + 1e-8)

    def normalize_pressure(self, pressure):
        """Normalize pressure to zero mean, unit variance"""
        mean_p = pressure.mean()
        std_p = pressure.std()
        return (pressure - mean_p) / (std_p + 1e-8)


def process_dataset(raw_dir, processed_dir, config):
    """
    Process entire dataset

    Args:
        raw_dir: Directory with raw CFD data
        processed_dir: Output directory for processed graphs
        config: Preprocessing configuration
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    converter = MeshToGraphConverter(config)

    # Find all VTK files
    vtk_files = list(raw_dir.glob('**/*.vtk')) + list(raw_dir.glob('**/*.vtu'))

    print(f"Found {len(vtk_files)} VTK files")

    data_list = []

    for vtk_file in tqdm(vtk_files, desc="Processing meshes"):
        try:
            mesh = converter.load_vtk_mesh(str(vtk_file))
            graph_data = converter.mesh_to_graph(mesh)
            data_list.append(graph_data)
        except Exception as e:
            print(f"Error processing {vtk_file}: {e}")
            continue

    # Save processed data
    print(f"\nSaving {len(data_list)} graphs to {processed_dir}")

    for i, data in enumerate(data_list):
        torch.save(data, processed_dir / f'graph_{i:05d}.pt')

    # Save metadata
    metadata = {
        'num_samples': len(data_list),
        'config': config,
        'source_dir': str(raw_dir)
    }

    with open(processed_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Preprocessing complete! Saved to {processed_dir}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess CFD data to graphs')
    parser.add_argument(
        '--raw_dir',
        type=str,
        default='data/raw',
        help='Directory with raw CFD data'
    )
    parser.add_argument(
        '--processed_dir',
        type=str,
        default='data/processed',
        help='Output directory for processed graphs'
    )
    parser.add_argument(
        '--edge_mode',
        type=str,
        choices=['radius', 'knn'],
        default='radius',
        help='Edge construction method'
    )
    parser.add_argument(
        '--edge_radius',
        type=float,
        default=0.1,
        help='Radius for edge connections'
    )
    parser.add_argument(
        '--max_neighbors',
        type=int,
        default=10,
        help='Maximum neighbors for KNN'
    )

    args = parser.parse_args()

    config = {
        'edge_mode': args.edge_mode,
        'edge_radius': args.edge_radius,
        'max_neighbors': args.max_neighbors
    }

    process_dataset(args.raw_dir, args.processed_dir, config)


if __name__ == '__main__':
    main()
