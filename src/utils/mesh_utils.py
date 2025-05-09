import numpy as np

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

def detect_dimension(mesh):
    print('mesh.utils.detect_dimension needs work')
    return 3
