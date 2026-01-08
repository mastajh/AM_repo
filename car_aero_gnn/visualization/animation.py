"""
Animation Generation for Unsteady Flow

Create videos and GIFs from flow trajectories
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm


try:
    import pyvista as pv
    import imageio
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False
    print("Warning: PyVista or imageio not installed")


def create_velocity_animation(
    trajectory,
    mesh,
    output_path='animation.gif',
    fps=30,
    cmap='jet'
):
    """
    Create animation of velocity field evolution

    Args:
        trajectory: List of states (PyG Data objects)
        mesh: PyVista mesh
        output_path: Output file path (.gif or .mp4)
        fps: Frames per second
        cmap: Colormap
    """
    if not HAS_LIBS:
        raise ImportError("PyVista and imageio required")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine output format
    if output_path.suffix == '.gif':
        use_gif = True
    elif output_path.suffix == '.mp4':
        use_gif = False
    else:
        raise ValueError("Output must be .gif or .mp4")

    # Compute global color limits
    all_vel_mag = []
    for state in trajectory:
        velocity = state.x[:, 3:6].numpy()
        vel_mag = np.linalg.norm(velocity, axis=1)
        all_vel_mag.append(vel_mag)

    vmin = min(v.min() for v in all_vel_mag)
    vmax = max(v.max() for v in all_vel_mag)

    # Generate frames
    frames = []

    for i, state in enumerate(tqdm(trajectory, desc="Rendering frames")):
        velocity = state.x[:, 3:6].numpy()
        vel_mag = np.linalg.norm(velocity, axis=1)

        mesh['velocity_magnitude'] = vel_mag

        plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
        plotter.add_mesh(
            mesh,
            scalars='velocity_magnitude',
            cmap=cmap,
            clim=[vmin, vmax],
            show_edges=False
        )
        plotter.add_scalar_bar(title='Velocity [m/s]', n_labels=5)
        plotter.add_text(
            f'Time step: {i}',
            position='upper_left',
            font_size=12
        )

        # Capture frame
        frame = plotter.screenshot(return_img=True)
        frames.append(frame)
        plotter.close()

    # Save animation
    if use_gif:
        imageio.mimsave(output_path, frames, fps=fps)
    else:
        imageio.mimsave(output_path, frames, fps=fps, codec='libx264')

    print(f"Animation saved to {output_path}")


def create_streamline_animation(
    trajectory,
    mesh,
    output_path='streamlines.mp4',
    fps=30,
    n_points=50
):
    """
    Create animation of evolving streamlines

    Args:
        trajectory: List of states
        mesh: PyVista mesh
        output_path: Output path
        fps: Frames per second
        n_points: Number of streamline seed points
    """
    if not HAS_LIBS:
        raise ImportError("Required libraries not available")

    frames = []

    for i, state in enumerate(tqdm(trajectory, desc="Rendering streamlines")):
        velocity = state.x[:, 3:6].numpy()
        mesh['velocity'] = velocity

        # Generate streamlines
        streamlines = mesh.streamlines(
            vectors='velocity',
            source_radius=0.5,
            n_points=n_points,
            max_steps=500
        )

        plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
        plotter.add_mesh(mesh, color='white', opacity=0.3)

        if streamlines.n_points > 0:
            plotter.add_mesh(
                streamlines.tube(radius=0.01),
                cmap='jet'
            )

        plotter.add_text(f'Time step: {i}', position='upper_left')

        frame = plotter.screenshot(return_img=True)
        frames.append(frame)
        plotter.close()

    imageio.mimsave(output_path, frames, fps=fps, codec='libx264')
    print(f"Streamline animation saved to {output_path}")


def create_comparison_animation(
    gnn_trajectory,
    cfd_trajectory,
    mesh,
    output_path='comparison.mp4',
    fps=30
):
    """
    Side-by-side comparison of GNN vs CFD

    Args:
        gnn_trajectory: GNN predictions
        cfd_trajectory: CFD ground truth
        mesh: PyVista mesh
        output_path: Output path
        fps: Frames per second
    """
    if not HAS_LIBS:
        raise ImportError("Required libraries not available")

    frames = []

    for i, (gnn_state, cfd_state) in enumerate(
        tqdm(
            zip(gnn_trajectory, cfd_trajectory),
            desc="Rendering comparison",
            total=len(gnn_trajectory)
        )
    ):
        gnn_vel = gnn_state.x[:, 3:6].numpy()
        cfd_vel = cfd_state.x[:, 3:6].numpy()

        gnn_mag = np.linalg.norm(gnn_vel, axis=1)
        cfd_mag = np.linalg.norm(cfd_vel, axis=1)

        vmin = min(gnn_mag.min(), cfd_mag.min())
        vmax = max(gnn_mag.max(), cfd_mag.max())

        mesh_gnn = mesh.copy()
        mesh_cfd = mesh.copy()
        mesh_gnn['vel'] = gnn_mag
        mesh_cfd['vel'] = cfd_mag

        plotter = pv.Plotter(
            shape=(1, 2),
            off_screen=True,
            window_size=[1920, 1080]
        )

        plotter.subplot(0, 0)
        plotter.add_mesh(mesh_gnn, scalars='vel', cmap='jet', clim=[vmin, vmax])
        plotter.add_text('GNN Prediction', position='upper_left')

        plotter.subplot(0, 1)
        plotter.add_mesh(mesh_cfd, scalars='vel', cmap='jet', clim=[vmin, vmax])
        plotter.add_text('CFD Ground Truth', position='upper_left')

        frame = plotter.screenshot(return_img=True)
        frames.append(frame)
        plotter.close()

    imageio.mimsave(output_path, frames, fps=fps, codec='libx264')
    print(f"Comparison animation saved to {output_path}")


if __name__ == '__main__':
    print("Animation module ready")
    print("Requires: pip install pyvista imageio imageio-ffmpeg")
