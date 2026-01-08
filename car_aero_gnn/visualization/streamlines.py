"""
Streamline Visualization

Generate and visualize streamlines for flow visualization
"""

import numpy as np
from pathlib import Path


try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


class StreamlineVisualizer:
    """
    Streamline generation and visualization

    Args:
        mesh: PyVista mesh with velocity data
    """

    def __init__(self, mesh):
        if not HAS_PYVISTA:
            raise ImportError("PyVista required")

        self.mesh = mesh

    def generate_streamlines(
        self,
        velocity,
        source_radius=0.5,
        n_points=100,
        max_steps=500,
        integration_direction='both'
    ):
        """
        Generate streamlines

        Args:
            velocity: [N, 3] velocity field
            source_radius: Radius of source points
            n_points: Number of seed points
            max_steps: Maximum integration steps
            integration_direction: 'forward', 'backward', or 'both'

        Returns:
            streamlines: PyVista streamline mesh
        """
        # Add velocity to mesh
        self.mesh['velocity'] = velocity

        # Generate streamlines
        streamlines = self.mesh.streamlines(
            vectors='velocity',
            source_radius=source_radius,
            n_points=n_points,
            max_steps=max_steps,
            integration_direction=integration_direction
        )

        return streamlines

    def visualize_streamlines(
        self,
        velocity,
        save_path=None,
        source_radius=0.5,
        n_points=100,
        tube_radius=0.01,
        cmap='jet',
        show_mesh=True
    ):
        """
        Visualize streamlines

        Args:
            velocity: Velocity field
            save_path: Save path
            source_radius: Source radius
            n_points: Number of seed points
            tube_radius: Tube radius for rendering
            cmap: Colormap
            show_mesh: Show underlying mesh
        """
        # Generate streamlines
        streamlines = self.generate_streamlines(
            velocity,
            source_radius=source_radius,
            n_points=n_points
        )

        # Create plotter
        plotter = pv.Plotter(off_screen=(save_path is not None))

        # Add mesh (semi-transparent)
        if show_mesh:
            plotter.add_mesh(
                self.mesh,
                color='white',
                opacity=0.3,
                show_edges=False
            )

        # Add streamlines as tubes
        if tube_radius > 0:
            streamline_tubes = streamlines.tube(radius=tube_radius)
            plotter.add_mesh(
                streamline_tubes,
                cmap=cmap,
                lighting=True
            )
        else:
            plotter.add_mesh(
                streamlines,
                cmap=cmap,
                line_width=2
            )

        plotter.add_scalar_bar(title='Velocity Magnitude [m/s]')

        if save_path:
            plotter.screenshot(save_path)
            plotter.close()
        else:
            plotter.show()

    def visualize_surface_streamlines(
        self,
        velocity,
        save_path=None,
        n_points=50
    ):
        """
        Visualize streamlines on surface

        Args:
            velocity: Velocity field
            save_path: Save path
            n_points: Number of streamlines
        """
        self.mesh['velocity'] = velocity

        # Extract surface
        surface = self.mesh.extract_surface()

        # Generate streamlines on surface
        streamlines = surface.streamlines(
            vectors='velocity',
            n_points=n_points,
            surface_streamlines=True
        )

        plotter = pv.Plotter(off_screen=(save_path is not None))

        plotter.add_mesh(surface, color='lightgray', opacity=0.8)
        plotter.add_mesh(streamlines.tube(radius=0.005), cmap='jet')

        if save_path:
            plotter.screenshot(save_path)
            plotter.close()
        else:
            plotter.show()


if __name__ == '__main__':
    print("Streamline visualizer ready")
