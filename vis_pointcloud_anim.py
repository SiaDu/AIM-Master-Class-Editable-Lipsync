import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import animation
from typing import Sequence, Union, Tuple, List

Array = Union[np.ndarray, Sequence[np.ndarray]]

# -----------------------------------------------------------------------------
# Internal helper: configure a 3D axis
# -----------------------------------------------------------------------------
def _setup_ax(
    ax: plt.Axes,
    title: str,
    elev: float,
    azim: float,
    limits: Tuple[float, float] = (-0.1, 0.1)
) -> None:
    ax.set_axis_off()
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_zlim(limits)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)

# -----------------------------------------------------------------------------
# Animate a single pointcloud sequence
# -----------------------------------------------------------------------------

def animate_pointcloud(
    verts: Array,
    title: str = "Frame",
    num_frames: int = 100,
    interval: int = 100,
    zoom: float = 1.0,
    elev: float = -90,
    azim: float = 90,
    point_size: float = 0.3,
) -> animation.FuncAnimation:
    """
    Animate one sequence of pointclouds in 3D.

    Parameters
    ----------
    verts : array or list of arrays, shape (T, V, 3)
    title : str, title prefix for each frame
    ...
    """
    arr = np.stack(verts, axis=0) if isinstance(verts, (list, tuple)) else verts
    arr = arr * zoom
    T = min(num_frames, arr.shape[0])

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    _setup_ax(ax, title, elev, azim)
    scat = ax.scatter([], [], [], s=point_size)

    def update(i):
        v = arr[i]
        scat._offsets3d = (v[:,0], v[:,1], v[:,2])
        ax.set_title(f"{title} Frame {i}")
        return scat,

    ani = animation.FuncAnimation(
        fig, update, frames=T, interval=interval, blit=False
    )
    plt.close(fig)
    return ani

# -----------------------------------------------------------------------------
# Animate comparison: two side-by-side pointcloud sequences
# -----------------------------------------------------------------------------

def animate_comparison(
    pred: Array,
    gt: Array,
    titles: Tuple[str, str] = ("Prediction", "Ground Truth"),
    num_frames: Union[int, None] = None,
    interval: int = 100,
    zoom: float = 1.0,
    elev: float = -90,
    azim: float = 90,
    point_size: float = 0.3,
) -> animation.FuncAnimation:
    """
    Side-by-side animation of pred vs gt.

    Parameters
    ----------
    pred, gt : arrays or lists, shape (T, V, 3)
    titles : tuple of two strings, panel titles
    ...
    """
    p = np.stack(pred, axis=0) if isinstance(pred, (list, tuple)) else pred
    g = np.stack(gt, axis=0)   if isinstance(gt,   (list, tuple)) else gt
    p *= zoom; g *= zoom
    T = min(p.shape[0], g.shape[0])
    if num_frames is None or num_frames > T:
        num_frames = T

    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    _setup_ax(ax1, titles[0], elev, azim)
    _setup_ax(ax2, titles[1], elev, azim)
    scat1 = ax1.scatter([], [], [], s=point_size)
    scat2 = ax2.scatter([], [], [], s=point_size)

    def update(i):
        v1, v2 = p[i], g[i]
        scat1._offsets3d = (v1[:,0], v1[:,1], v1[:,2])
        scat2._offsets3d = (v2[:,0], v2[:,1], v2[:,2])
        ax1.set_title(f"{titles[0]} Frame {i}")
        ax2.set_title(f"{titles[1]} Frame {i}")
        return scat1, scat2

    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=interval, blit=False
    )
    plt.close(fig)
    return ani

# -----------------------------------------------------------------------------
# Animate triple: three side-by-side pointcloud sequences
# -----------------------------------------------------------------------------

def animate_triple(
    pred: Array,
    neutral: Array,
    gt: Array,
    titles: Tuple[str, str, str] = ("Prediction", "Neutral", "Ground Truth"),
    num_frames: Union[int, None] = None,
    interval: int = 100,
    zoom: float = 1.0,
    elev: float = -90,
    azim: float = 90,
    point_size: float = 0.3,
) -> animation.FuncAnimation:
    """
    Three-way side-by-side animation.

    Parameters
    ----------
    pred, neutral, gt : arrays or lists, shape (T, V, 3)
    titles : tuple of three strings
    ...
    """
    p = np.stack(pred, axis=0)    if isinstance(pred,    (list, tuple)) else pred
    n = np.stack(neutral, axis=0) if isinstance(neutral, (list, tuple)) else neutral
    g = np.stack(gt, axis=0)      if isinstance(gt,      (list, tuple)) else gt
    p*=zoom; n*=zoom; g*=zoom
    T = min(p.shape[0], n.shape[0], g.shape[0])
    if num_frames is None or num_frames > T:
        num_frames = T

    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1,3,1, projection='3d')
    ax2 = fig.add_subplot(1,3,2, projection='3d')
    ax3 = fig.add_subplot(1,3,3, projection='3d')
    _setup_ax(ax1, titles[0], elev, azim)
    _setup_ax(ax2, titles[1], elev, azim)
    _setup_ax(ax3, titles[2], elev, azim)
    scat1 = ax1.scatter([], [], [], s=point_size)
    scat2 = ax2.scatter([], [], [], s=point_size)
    scat3 = ax3.scatter([], [], [], s=point_size)

    def update(i):
        v1, v2, v3 = p[i], n[i], g[i]
        scat1._offsets3d = (v1[:,0], v1[:,1], v1[:,2])
        scat2._offsets3d = (v2[:,0], v2[:,1], v2[:,2])
        scat3._offsets3d = (v3[:,0], v3[:,1], v3[:,2])
        ax1.set_title(f"{titles[0]} Frame {i}")
        ax2.set_title(f"{titles[1]} Frame {i}")
        ax3.set_title(f"{titles[2]} Frame {i}")
        return scat1, scat2, scat3

    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=interval, blit=False
    )
    plt.close(fig)
    return ani
