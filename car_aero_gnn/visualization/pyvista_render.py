"""
3D Visualization using PyVista

Renders velocity fields, pressure distributions, and flow features
"""

import numpy as np
from pathlib import Path


try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("PyVista not installed. Install with: pip install pyvista")


class AeroVisualizer:
    """
    3D flow field visualization

    Args:
        mesh: PyVista mesh or path to mesh file
    """

    def __init__(self, mesh):
        if not HAS_PYVISTA:
            raise ImportError("PyVista required for visualization")

        if isinstance(mesh, (str, Path)):
            self.mesh = pv.read(mesh)
        else:
            self.mesh = mesh

        self.plotter = None

    def visualize_velocity_magnitude(
        self,
        velocity,
        save_path=None,
        cmap='jet',
        show_edges=False,
        camera_position='xy'
    ):
        """
        Visualize velocity magnitude

        Args:
            velocity: [N, 3] velocity field
            save_path: Path to save image (optional)
            cmap: Colormap name
            show_edges: Show mesh edges
            camera_position: Camera view
        """
        # Compute velocity magnitude
        vel_mag = np.linalg.norm(velocity, axis=1)
        self.mesh['velocity_magnitude'] = vel_mag

        # Create plotter
        plotter = pv.Plotter(off_screen=(save_path is not None))

        # Add mesh
        plotter.add_mesh(
            self.mesh,
            scalars='velocity_magnitude',
            cmap=cmap,
            clim=[0, vel_mag.max()],
            show_edges=show_edges,
            lighting=True
        )

        # Add scalar bar
        plotter.add_scalar_bar(
            title='Velocity Magnitude [m/s]',
            n_labels=5,
            italic=False,
            fmt='%.2f'
        )

        # Set camera
        plotter.camera_position = camera_position

        # Show or save
        if save_path:
            plotter.screenshot(save_path, transparent_background=False)
            print(f"Saved to {save_path}")
            plotter.close()
        else:
            plotter.show()

    def visualize_pressure(
        self,
        pressure,
        save_path=None,
        coefficient=True,
        p_inf=0.0,
        q_inf=1.0,
        cmap='coolwarm'
    ):
        """
        Visualize pressure distribution

        Args:
            pressure: [N] pressure field
            save_path: Save path
            coefficient: Show pressure coefficient instead of absolute pressure
            p_inf: Freestream pressure
            q_inf: Dynamic pressure (0.5 * rho * U^2)
            cmap: Colormap
        """
        if coefficient:
            # Compute pressure coefficient
            cp = (pressure - p_inf) / q_inf
            self.mesh['Cp'] = cp
            scalar_name = 'Cp'
            title = 'Pressure Coefficient'
            clim = [-1.5, 1.0]
        else:
            self.mesh['pressure'] = pressure
            scalar_name = 'pressure'
            title = 'Pressure [Pa]'
            clim = [pressure.min(), pressure.max()]

        # Create plotter
        plotter = pv.Plotter(off_screen=(save_path is not None))

        # Add mesh
        plotter.add_mesh(
            self.mesh,
            scalars=scalar_name,
            cmap=cmap,
            clim=clim,
            show_edges=False
        )

        plotter.add_scalar_bar(title=title, n_labels=5)

        # Show or save
        if save_path:
            plotter.screenshot(save_path)
            print(f"Saved to {save_path}")
            plotter.close()
        else:
            plotter.show()

    def visualize_velocity_components(
        self,
        velocity,
        save_path=None,
        component='u'
    ):
        """
        Visualize individual velocity component

        Args:
            velocity: [N, 3] velocity field
            save_path: Save path
            component: 'u', 'v', or 'w'
        """
        component_idx = {'u': 0, 'v': 1, 'w': 2}[component]
        component_name = {'u': 'U_x', 'v': 'U_y', 'w': 'U_z'}[component]

        self.mesh[component_name] = velocity[:, component_idx]

        plotter = pv.Plotter(off_screen=(save_path is not None))

        plotter.add_mesh(
            self.mesh,
            scalars=component_name,
            cmap='RdBu',
            show_edges=False
        )

        plotter.add_scalar_bar(title=f'{component_name} [m/s]')

        if save_path:
            plotter.screenshot(save_path)
            plotter.close()
        else:
            plotter.show()

    def visualize_slices(
        self,
        velocity,
        pressure,
        slice_positions,
        save_path=None
    ):
        """
        Visualize field on multiple slice planes

        Args:
            velocity: Velocity field
            pressure: Pressure field
            slice_positions: List of (axis, position) tuples
            save_path: Save path
        """
        vel_mag = np.linalg.norm(velocity, axis=1)
        self.mesh['velocity_magnitude'] = vel_mag
        self.mesh['pressure'] = pressure

        plotter = pv.Plotter(off_screen=(save_path is not None))

        # Add semi-transparent surface
        plotter.add_mesh(
            self.mesh,
            opacity=0.1,
            color='white'
        )

        # Add slices
        for axis, position in slice_positions:
            if axis == 'x':
                slice_mesh = self.mesh.slice(normal='x', origin=[position, 0, 0])
            elif axis == 'y':
                slice_mesh = self.mesh.slice(normal='y', origin=[0, position, 0])
            else:  # z
                slice_mesh = self.mesh.slice(normal='z', origin=[0, 0, position])

            plotter.add_mesh(
                slice_mesh,
                scalars='velocity_magnitude',
                cmap='jet'
            )

        plotter.add_scalar_bar(title='Velocity Magnitude [m/s]')

        if save_path:
            plotter.screenshot(save_path)
            plotter.close()
        else:
            plotter.show()

    def create_multi_view(
        self,
        velocity,
        pressure,
        save_path=None
    ):
        """
        Create multi-view visualization

        Args:
            velocity: Velocity field
            pressure: Pressure field
            save_path: Save path
        """
        vel_mag = np.linalg.norm(velocity, axis=1)
        self.mesh['velocity_magnitude'] = vel_mag
        self.mesh['pressure'] = pressure

        # Create subplot plotter
        plotter = pv.Plotter(
            shape=(1, 2),
            off_screen=(save_path is not None)
        )

        # Velocity view
        plotter.subplot(0, 0)
        plotter.add_mesh(
            self.mesh,
            scalars='velocity_magnitude',
            cmap='jet'
        )
        plotter.add_text('Velocity Magnitude', position='upper_left')

        # Pressure view
        plotter.subplot(0, 1)
        plotter.add_mesh(
            self.mesh,
            scalars='pressure',
            cmap='coolwarm'
        )
        plotter.add_text('Pressure', position='upper_left')

        if save_path:
            plotter.screenshot(save_path)
            plotter.close()
        else:
            plotter.show()


def visualize_comparison(
    mesh,
    prediction,
    ground_truth,
    save_path=None
):
    """
    Side-by-side comparison of prediction vs ground truth

    Args:
        mesh: PyVista mesh
        prediction: Predicted field
        ground_truth: Ground truth field
        save_path: Save path
    """
    if not HAS_PYVISTA:
        raise ImportError("PyVista required")

    mesh_pred = mesh.copy()
    mesh_gt = mesh.copy()

    pred_mag = np.linalg.norm(prediction, axis=1)
    gt_mag = np.linalg.norm(ground_truth, axis=1)

    mesh_pred['field'] = pred_mag
    mesh_gt['field'] = gt_mag

    # Shared color limits
    vmin = min(pred_mag.min(), gt_mag.min())
    vmax = max(pred_mag.max(), gt_mag.max())

    plotter = pv.Plotter(shape=(1, 2), off_screen=(save_path is not None))

    plotter.subplot(0, 0)
    plotter.add_mesh(mesh_pred, scalars='field', cmap='jet', clim=[vmin, vmax])
    plotter.add_text('GNN Prediction', position='upper_left')

    plotter.subplot(0, 1)
    plotter.add_mesh(mesh_gt, scalars='field', cmap='jet', clim=[vmin, vmax])
    plotter.add_text('Ground Truth (CFD)', position='upper_left')

    if save_path:
        plotter.screenshot(save_path)
        plotter.close()
    else:
        plotter.show()


if __name__ == '__main__':
    print("PyVista visualizer ready")
    print("Requires: pip install pyvista")
