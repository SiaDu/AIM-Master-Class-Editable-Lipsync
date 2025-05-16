import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import imageio
from scipy.io import wavfile
import subprocess

def _set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range) / 2.0
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)

def render_mesh_video(
    verts_seq: np.ndarray,    # (T, V, 3)
    faces: np.ndarray,        # (F, 3)
    audio_path: str,
    video_path: str   = "mesh.mp4",
    final_path: str   = "final.mp4",
    fps: float        = None,
    figsize: tuple    = (6,6),
    elev: float       = 90,
    azim: float       = -90
):
    """
    Use matplotlib plot_trisurf to render the mesh and synthesize the video with audio.
    """
    # 1) Read audio, calculate duration and frame rate
    sr, audio = wavfile.read(audio_path)
    duration = audio.shape[0] / sr
    T = verts_seq.shape[0]
    fps = fps or (T / duration)

    # 2) Initialize the video writer
    writer = imageio.get_writer(video_path, fps=fps, codec='libx264')

    # 3) Render and write per frame
    for i in range(T):
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection='3d')
        verts = verts_seq[i]

        ax.plot_trisurf(
            verts[:,0], verts[:,1], verts[:,2],
            triangles=faces,
            cmap="viridis",
            linewidth=0.2,
            antialiased=True,
            shade=True
        )
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        _set_axes_equal(ax)           # ← Keep x/y/z in equal proportion
        plt.tight_layout()

        # Fetching image data from the canvas
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.append_data(frame)
        plt.close(fig)

    writer.close()

    # 4) Merge audio and video
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", final_path
    ], check=True)

    print(f"► mesh video saved as `{video_path}`, final w/ audio as `{final_path}`")

# —— Usage Example ——#
# from render_mesh_video import render_mesh_video
# pred_v = np.load('pred_vertices.npy')        # (T,5023,3)
# faces  = np.load('faces.npy')                # (F,3)
# render_mesh_video(
#     verts_seq = pred_v,
#     faces     = faces,
#     audio_path= 'test_sentence.wav',
#     video_path= 'mesh.mp4',
#     final_path= 'final_with_audio.mp4',
#     fps       = None,
#     figsize   = (6,6),
#     elev      = 90,
#     azim      = -90
# )
