"""
Wake Analysis and Visualization

Analyze and visualize wake structures behind vehicles
"""

import numpy as np
from pathlib import Path


try:
    import pyvista as pv
    import matplotlib.pyplot as plt
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False


class WakeAnalyzer:
    """
    Analyze wake behind vehicle

    Args:
        mesh: PyVista mesh
        velocity: Velocity field
        car_length: Vehicle length for normalization
    """

    def __init__(self, mesh, velocity, car_length=4.5):
        if not HAS_LIBS:
            raise ImportError("PyVista and matplotlib required")

        self.mesh = mesh
        self.velocity = velocity
        self.L = car_length

        # Add velocity to mesh
        self.mesh['velocity'] = velocity

    def extract_wake_plane(self, x_distance):
        """
        Extract cross-sectional plane in wake

        Args:
            x_distance: Distance behind vehicle (in car lengths)

        Returns:
            plane: Sliced mesh at specified location
        """
        plane = self.mesh.slice(
            normal='x',
            origin=[x_distance * self.L, 0, 0]
        )

        return plane

    def compute_velocity_deficit(
        self,
        x_distances=[0.5, 1.0, 2.0, 3.0],
        save_path=None
    ):
        """
        Compute and visualize velocity deficit at various distances

        Args:
            x_distances: List of distances (in car lengths)
            save_path: Path to save figure
        """
        U_inf = np.max(np.linalg.norm(self.velocity, axis=1))

        fig, axes = plt.subplots(
            1, len(x_distances),
            figsize=(4 * len(x_distances), 4)
        )

        if len(x_distances) == 1:
            axes = [axes]

        for ax, x_dist in zip(axes, x_distances):
            plane = self.extract_wake_plane(x_dist)

            if plane.n_points > 0:
                vel_mag = np.linalg.norm(plane['velocity'], axis=1)
                deficit = 1 - vel_mag / U_inf

                y = plane.points[:, 1] / self.L
                z = plane.points[:, 2] / self.L

                sc = ax.scatter(y, z, c=deficit, cmap='hot', s=10, vmin=0, vmax=1)
                ax.set_title(f'x/L = {x_dist}')
                ax.set_xlabel('y/L')
                ax.set_ylabel('z/L')
                ax.set_aspect('equal')
                plt.colorbar(sc, ax=ax, label='Velocity Deficit')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def compute_q_criterion(self):
        """
        Compute Q-criterion for vortex identification

        Q = 0.5 * (||Ω||² - ||S||²)

        Returns:
            Q: Q-criterion values at each node
        """
        # This is a simplified placeholder
        # Proper implementation requires computing velocity gradients
        print("Q-criterion computation requires velocity gradient calculation")
        print("This is a simplified version")

        # Placeholder: use velocity magnitude variation as proxy
        vel_mag = np.linalg.norm(self.velocity, axis=1)
        Q = np.abs(vel_mag - vel_mag.mean())

        return Q

    def visualize_vortex_structures(
        self,
        q_threshold=None,
        save_path=None
    ):
        """
        Visualize vortex structures using Q-criterion

        Args:
            q_threshold: Threshold for Q-criterion
            save_path: Save path
        """
        Q = self.compute_q_criterion()
        self.mesh['Q'] = Q

        if q_threshold is None:
            q_threshold = np.percentile(Q, 90)

        # Extract isosurface
        contour = self.mesh.contour([q_threshold], scalars='Q')

        plotter = pv.Plotter(off_screen=(save_path is not None))

        # Add vehicle surface (semi-transparent)
        surface = self.mesh.extract_surface()
        plotter.add_mesh(surface, color='gray', opacity=0.2)

        # Add vortex structures
        if contour.n_points > 0:
            vel_mag = np.linalg.norm(contour['velocity'], axis=1)
            contour['vel_mag'] = vel_mag

            plotter.add_mesh(
                contour,
                scalars='vel_mag',
                cmap='jet',
                opacity=0.8
            )

        plotter.add_text(f'Q-criterion threshold = {q_threshold:.2f}')

        if save_path:
            plotter.screenshot(save_path)
            plotter.close()
        else:
            plotter.show()

    def analyze_recirculation_zone(self):
        """
        Detect and analyze recirculation zones

        Returns:
            recirculation_info: Dictionary with recirculation zone properties
        """
        # Simplified: detect regions with negative u-velocity
        u_velocity = self.velocity[:, 0]
        recirculation_mask = u_velocity < 0

        num_recirculation_nodes = np.sum(recirculation_mask)
        total_nodes = len(u_velocity)
        recirculation_fraction = num_recirculation_nodes / total_nodes

        info = {
            'num_nodes': num_recirculation_nodes,
            'fraction': recirculation_fraction,
            'min_u_velocity': u_velocity[recirculation_mask].min() if num_recirculation_nodes > 0 else 0
        }

        return info


if __name__ == '__main__':
    print("Wake analyzer ready")
