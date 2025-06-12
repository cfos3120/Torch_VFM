import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_normal_vector(ax, start_coord, normal, color='red'):
    # Plot normal vector (arrow from center)
    start = start_coord.numpy()
    vector = normal.numpy()
    ax.quiver(*start, *vector, length=0.006, normalize=True, color=color, linewidth=2)
    return ax

def plot_face_3d(ax, points, color='lightgreen', elev=30, azim=-60, show_vertices=True):
    """
    Plots a single polygonal face (3 or more points) in 3D using matplotlib.

    Args:
        points (list or tensor): List or tensor of shape (N, 3) with vertex coordinates.
        color (str): Face color.
        elev (float): Elevation angle for 3D view.
        azim (float): Azimuthal angle for 3D view.
        show_vertices (bool): Whether to plot vertex dots.
    """
    # Convert input to numpy array
    if isinstance(points, torch.Tensor):
        points = points.numpy()
    else:
        points = np.array(points)

    if points.shape[0] < 3 or points.shape[1] != 3:
        raise ValueError("Input must be an Nx3 array with at least 3 points.")

    # Plot polygon face
    poly = Poly3DCollection([points], alpha=0.5, facecolor=color, edgecolor='k')
    ax.add_collection3d(poly)

    # Optional: Plot vertex dots
    if show_vertices:
        ax.scatter(*zip(*points), color='black', s=50)

    # Set axis limits with margin
    margin = 0.05
    min_vals = points.min(axis=0) - margin
    max_vals = points.max(axis=0) + margin
    ax.set_xlim([min_vals[0], max_vals[0]])
    ax.set_ylim([min_vals[1], max_vals[1]])
    ax.set_zlim([min_vals[2], max_vals[2]])

    # Labels and view
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Polygon Face")
    ax.view_init(elev=elev, azim=azim)

    return ax

def project_to_2d(points):
    """
    Projects 3D points to 2D by dropping the dimension with least variation.
    """
    ranges = points.max(axis=0) - points.min(axis=0)
    drop_dim = np.argmin(ranges)
    keep_dims = [i for i in range(3) if i != drop_dim]
    return points[:, keep_dims]

def segments_intersect(p1, p2, q1, q2):
    """
    Checks if 2 segments (p1-p2 and q1-q2) intersect in 2D using orientation.
    """

    def orientation(a, b, c):
        # Cross product to find orientation
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, c):
        # Check if point b lies on segment a-c
        return min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    # General case
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    # Special cases (colinear and overlapping)
    if o1 == 0 and on_segment(p1, q1, p2): return True
    if o2 == 0 and on_segment(p1, q2, p2): return True
    if o3 == 0 and on_segment(q1, p1, q2): return True
    if o4 == 0 and on_segment(q1, p2, q2): return True

    return False

def check_quad_is_valid(vertices):
    """
    Checks whether a 4-point polygon is a valid (non-intersecting) quadrilateral.

    Args:
        vertices (torch.Tensor): (4, 3) tensor of 3D points.

    Returns:
        bool: True if the quad does not self-intersect, False otherwise.
    """
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices)

    if vertices.shape != (4, 3):
        raise ValueError("Input must be a (4, 3) tensor of 3D points.")

    # Project to 2D
    projected = project_to_2d(vertices.numpy())

    # Check diagonals (edges 0-1, 1-2, 2-3, 3-0)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    # Check for cross intersection: (0-1 with 2-3) and (1-2 with 3-0)
    if segments_intersect(projected[0], projected[1], projected[2], projected[3]):
        return False
    if segments_intersect(projected[1], projected[2], projected[3], projected[0]):
        return False

    return True