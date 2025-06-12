import numpy as np

def check_vector_alignment(vectors_a, vectors_b):
    aligned = []
    for a, b in zip(vectors_a, vectors_b):
        a = np.array(a)
        b = np.array(b)
        dot = np.dot(a, b)
        aligned.append(dot > 0)
    return aligned

def plane_equation_from_points(p1, p2, p3):
    """
    Compute the plane equation Ax + By + Cz + D = 0 from three points.
    Returns A, B, C, D.
    """
    # Two vectors in the plane
    v1 = p3 - p1
    v2 = p2 - p1
    
    # Normal vector is the cross product of v1 and v2
    normal = np.cross(v1, v2)
    A, B, C = normal
    D = -np.dot(normal, p1)
    return A, B, C, D

def find_indices_dict(A, B):
    # Convert coordinates to tuples for hashing
    coord_map = {tuple(coord): idx for idx, coord in enumerate(B)}
    return [coord_map[tuple(coord)] for coord in A if tuple(coord) in coord_map]

def compute_face_normal(face_points, points):
    verts = points[list(face_points)]
    if len(verts) < 3:
        return np.array([0.0, 0.0, 0.0])
    center = verts.mean(axis=0)
    normal = np.zeros(3)
    for i in range(len(verts)):
        v1 = verts[i] - center
        v2 = verts[(i + 1) % len(verts)] - center
        normal += np.cross(v1, v2)
    normal /= 2
    norm = np.linalg.norm(normal)
    return normal / norm if norm > 0 else np.zeros(3)

def compute_face_area(face_pts, points):
    verts = points[list(face_pts)]
    if len(verts) < 3:
        return 0.0
    center = verts.mean(axis=0)
    area = 0.0
    for i in range(len(verts)):
        v1 = verts[i] - center
        v2 = verts[(i + 1) % len(verts)] - center
        area += np.linalg.norm(np.cross(v1, v2)) / 2
    return area

def detect_dimension(cell_coordinates):
    spatial_dims_list = np.all(cell_coordinates == cell_coordinates[0, :], axis=0)
    spatial_dims_list = np.where(~spatial_dims_list)[0].tolist()
    if len(spatial_dims_list) == 2:
        print('Excluding Z-dim in mesh')
    """
    TODO: Return the index list instead, and cut off the dims, to handle meshes, where Y or X is the empty dim
    """
    return len(spatial_dims_list)

def compute_true_geometric_centroid(mesh):
    n_cells = mesh.n_cells
    cell_centres = []
    for cell_id in range(n_cells):
        cell = mesh.GetCell(cell_id)
        point_ids = [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]
        points = np.array([mesh.GetPoint(pid) for pid in point_ids])

        # Hexahedron decomposition into 5 tetrahedra (there are multiple valid ways)
        # Each row defines a tetrahedron by indices into the `points` array
        tetrahedra = [
            [0, 1, 3, 4],
            [1, 2, 3, 6],
            [1, 4, 5, 6],
            [3, 4, 6, 7],
            [1, 3, 4, 6],
        ]

        volumes = []
        centroids = []

        for tet in tetrahedra:
            a, b, c, d = [points[i] for i in tet]
            vol = np.abs(np.dot(np.cross(b - a, c - a), d - a)) / 6.0
            centroid = (a + b + c + d) / 4.0
            volumes.append(vol)
            centroids.append(centroid)

        volumes = np.array(volumes)
        centroids = np.array(centroids)
        total_volume = np.sum(volumes)

        if total_volume == 0:
            cell_centres.append(np.mean(points, axis=0))  # fallback: average of corners
        else:
            cell_centres.append(np.average(centroids, axis=0, weights=volumes))
    return np.array(cell_centres)