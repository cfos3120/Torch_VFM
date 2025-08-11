import torch
import numpy as np
import pyvista as pv
from .utils.mesh_utils import *

class vfm_mesh_geometry():
    def __init__(self, mesh, device='cpu', dtype=torch.float32) -> None:
        self.device = device
        self.dtype = dtype
        self.dim = 3    # TODO: we can adjust for 2D meshes later
        self.n_cells = mesh.n_cells
        self.n_points = mesh.n_points
        self.vertices = torch.tensor(mesh.points, dtype=self.dtype, device=self.device)
        try:
            print('Trying to fetch cell centers and volume from mesh...')
            self.cell_centers = torch.tensor(mesh.cell_data['C'], dtype=self.dtype, device=self.device)
            self.cell_volumes = torch.tensor(mesh.cell_data['Vc'], dtype=self.dtype, device=self.device)
            print('Cell centers and Volumes fetched successfully')
        except Exception as e:
            print('An error was encountered when trying to read Cell and Volume files\n',e,
                  '\nManually calculating cell centers and volume for mesh...')
            self.cell_centers = torch.tensor(compute_true_geometric_centroid(mesh), dtype=self.dtype, device=self.device)
            self.cell_volumes = torch.tensor(mesh.compute_cell_sizes()["Volume"], dtype=self.dtype, device=self.device)

        self._fetch_vtk_faces(mesh)
        self._face_connectivity()
        
        self._calculate_face_centers_and_areas()
        self._connectivity_vectors()
        #self._correct_vector_direction()
        self._compute_skewness_and_orthogonality()

    def _fetch_vtk_faces(self, mesh) -> None:
        faces_dict = {}  # maps face_key -> face_index
        self.faces = []       # List[List[int]]: each face is a list of point indices
        self.cell_faces = []  # List[List[int]]: for each cell, a list of face indices
        
        # Loop over every cell in the mesh
        for cell_id in range(self.n_cells):
            vtk_cell = mesh.GetCell(cell_id)
            n_faces = vtk_cell.GetNumberOfFaces()
            cell_faces_indices = []

            # Loop over each face in the cell
            for face_id in range(n_faces):
                face = vtk_cell.GetFace(face_id)
                face_pts = [face.GetPointId(i) for i in range(face.GetNumberOfPoints())]
                key = tuple(sorted(face_pts))

                # Check if the face has already been seen, use the existing index
                if key not in faces_dict:
                    faces_dict[key] = len(self.faces)
                    self.faces.append(list(face_pts))
                cell_faces_indices.append(faces_dict[key])
            self.cell_faces.append(cell_faces_indices)
        
        self.num_faces = len(self.faces)

    def _correct_vector_direction(self) -> None:
        match_list = check_vector_alignment(self.face_areas, self.cell_center_unit_vectors)
        misaligned_indices = np.where(~np.array(match_list))[0]
        self.face_areas[misaligned_indices] *= -1  # Flip the sign for misaligned vectors
        self.face_normal_unit_vectors[misaligned_indices] *= -1 
        print(f"Corrected {len(misaligned_indices)} misaligned face area vectors")

    def _calculate_face_centers_and_areas(self):
        """Calculate face centers and area vectors."""
        # Debug print
        print(f"Calculating face centers and areas for {len(self.faces)} faces")
        
        # Initialize with correct size
        self.face_centers = torch.zeros((len(self.faces), 3), device=self.device, dtype=self.dtype)
        self.face_areas = torch.zeros((len(self.faces), 3), device=self.device, dtype=self.dtype)
        
        for i, face in enumerate(self.faces):
            # Get face vertices
            face_vertices = self.vertices[face]
            
            # Calculate face center (average of vertices)
            self.face_centers[i] = torch.mean(face_vertices, dim=0)
            
            # Calculate face area vector using triangulation
            # For simplicity, we assume planar faces
            v0 = face_vertices[0]
            n_vertices = len(face)
            area_vector = torch.zeros(3, device=self.device, dtype=self.dtype)
            
            for j in range(1, n_vertices - 1):
                v1 = face_vertices[j]
                v2 = face_vertices[j + 1]
                # Cross product for triangle area
                triangle_area  = 0.5 * torch.linalg.cross(v1 - v0, v2 - v0)
                area_vector += triangle_area#.abs() 
                
            self.face_areas[i] = area_vector
        
        self.face_areas_mag = torch.norm(self.face_areas, dim=-1, keepdim=False)
        self.face_normal_unit_vectors = torch.nn.functional.normalize(self.face_areas, dim=1)

    def _face_connectivity(self):
        """Determine face owners and neighbors."""
        # Initialize arrays
        self.face_owners = torch.zeros(self.num_faces, dtype=torch.int64, device=self.device)
        self.face_neighbors = torch.zeros(self.num_faces, dtype=torch.int64, device=self.device)
        self.face_neighbors.fill_(-1)  # -1 indicates boundary face (no neighbor)
        
        # Create face-to-cell mapping
        face_cells = [[] for _ in range(self.num_faces)]
        for cell_idx, cell_face_indices in enumerate(self.cell_faces):
            #print(cell_face_indices)
            for face_idx in cell_face_indices:
                face_cells[face_idx].append(cell_idx)

        # Assign owners and neighbors
        for face_idx, cells in enumerate(face_cells):
            if len(cells) == 1:
                # Boundary face
                self.face_owners[face_idx] = cells[0]
            elif len(cells) == 2:
                # Internal face
                # Determine owner and neighbor based on cell indices
                # By convention, the lower index is the owner
                if cells[0] < cells[1]:
                    self.face_owners[face_idx] = cells[0]
                    self.face_neighbors[face_idx] = cells[1]
                else:
                    self.face_owners[face_idx] = cells[1]
                    self.face_neighbors[face_idx] = cells[0]
            else:
                raise ValueError(f"Face {face_idx} is connected to {len(cells)} cells, expected 1 or 2")
        
        # Identify internal faces
        self.internal_faces = torch.where(self.face_neighbors >= 0)[0]
        self.boundary_faces = torch.where(self.face_neighbors == -1)[0]
        self.num_internal_faces = len(self.internal_faces)
        self.num_boundary_faces = len(self.boundary_faces)

    def _connectivity_vectors(self):
        """Calculate geometric properties needed for corrections."""
        # Vectors from owner to neighbor cell centers
        self.cell_center_vectors = torch.zeros((self.num_faces, 3), device=self.device, dtype=self.dtype)

        # For internal faces
        for i in range(self.num_internal_faces):
            face_idx = self.internal_faces[i]
            owner = self.face_owners[face_idx]
            neighbor = self.face_neighbors[face_idx]
            self.cell_center_vectors[face_idx] = self.cell_centers[neighbor] - self.cell_centers[owner]
        
        # For boundary faces
        for face_idx in range(self.num_faces):
            if self.face_neighbors[face_idx] < 0:
                owner = self.face_owners[face_idx]
                # Draw line from cell center to face center
                self.cell_center_vectors[face_idx] = (self.face_centers[face_idx] - self.cell_centers[owner]) * 2
        
        # Unit vectors
        self.cell_center_unit_vectors = torch.nn.functional.normalize(self.cell_center_vectors, dim=1)

    def _compute_skewness_and_orthogonality(self):
        # Calculate non-orthogonality
        self.orthogonality = torch.abs(torch.sum(self.cell_center_unit_vectors * self.face_normal_unit_vectors, dim=1))
        self.non_orthogonality_angle = torch.acos(torch.clamp(self.orthogonality, -1.0, 1.0)) * (180.0 / torch.pi)
        
        # Calculate skewness vectors
        self.skewness_vectors = torch.zeros((self.num_faces, 3), device=self.device)
        for face_idx in range(self.num_faces):
            owner = self.face_owners[face_idx]
            
            if self.face_neighbors[face_idx] >= 0:
                # Internal face
                neighbor = self.face_neighbors[face_idx]
                
                # Vector from owner to neighbor
                d = self.cell_centers[neighbor] - self.cell_centers[owner]
                d_mag = torch.norm(d)
                d_unit = d / d_mag
                
                # Face center
                fc = self.face_centers[face_idx]
                
                # Project face center onto line connecting cell centers
                p_owner = self.cell_centers[owner]
                t = torch.dot(fc - p_owner, d_unit)
                intersection = p_owner + t * d_unit
                
                # Skewness vector: from face center to intersection point
                self.skewness_vectors[face_idx] = intersection - fc
            else:
                # Boundary face - simplified approach
                # For boundaries, we use a reflection of the owner cell
                self.skewness_vectors[face_idx] = torch.zeros(3, device=self.device)

    def to(self, device):
        """Move mesh data to specified device."""
        self.device = device
        self.vertices = self.vertices.to(device)
        self.face_centers = self.face_centers.to(device)
        self.face_areas = self.face_areas.to(device)
        self.cell_centers = self.cell_centers.to(device)
        self.cell_volumes = self.cell_volumes.to(device)
        self.face_owners = self.face_owners.to(device)
        self.face_neighbors = self.face_neighbors.to(device)
        self.internal_faces = self.internal_faces.to(device)
        self.cell_center_vectors = self.cell_center_vectors.to(device)
        self.cell_center_unit_vectors = self.cell_center_unit_vectors.to(device)
        self.face_normal_unit_vectors = self.face_normal_unit_vectors.to(device)
        self.orthogonality = self.orthogonality.to(device)
        self.non_orthogonality_angle = self.non_orthogonality_angle.to(device)
        self.skewness_vectors = self.skewness_vectors.to(device)
        return self

    def describe(self):
        print('Mesh Geometry:')
        print(f'- n_cells {self.n_cells}')
        print(f'- n_points {self.n_points}')
        print(f'- points {self.vertices.shape}')
        print(f'- cell_centres {self.cell_centers.shape}')
        print(f'- cell_faces {len(self.cell_faces)}')
        print(f'- faces {len(self.faces)}')
        print(f'- internal_faces {self.num_internal_faces}')
        print(f'- boundary_faces {self.num_boundary_faces}')

    def _add_boundaries(self, boundaries:dict) -> None:
        """
        Add boundary patches to the mesh geometry.
        
        Args:
            boundaries: Dictionary of boundary patches. patches are in vtk format.
        """
        # Create a dictionary to map patch names to face indices
        self.patch_face_keys = dict.fromkeys(boundaries.keys())
        # Loop through each patch
        total_faces_found = 0
        for patch in self.patch_face_keys:

            # Find the indices of the faces that belong to this patch
            patch_point_indices = find_indices_dict(boundaries[patch].points, self.vertices.cpu().numpy())
            
            patch_face_idx= []
            for face_key in self.boundary_faces:
                if np.all(np.isin([x for x in self.faces[face_key]], patch_point_indices)):
                    patch_face_idx.append(face_key)
            if patch_face_idx == []:
                raise KeyError(f'Patch {patch} was not found in the face list')
            else:
                print(f' Found Patch "{patch}" with {len(patch_face_idx)} Faces')
                total_faces_found += len(patch_face_idx)
            
            self.patch_face_keys[patch] = torch.tensor(patch_face_idx, dtype=torch.int64)
            
        print(f'Boundary faces indexed: {total_faces_found}/{len(self.boundary_faces)} patches found')

class MeshQuality:
    """
    Class for calculating and reporting mesh quality metrics.
    """
    
    @staticmethod
    def calculate_non_orthogonality(mesh: vfm_mesh_geometry) -> torch.Tensor:
        """
        Calculate non-orthogonality angle for each face.
        
        Args:
            mesh: MeshGeometry object
            
        Returns:
            Tensor of shape [num_faces] containing non-orthogonality angles in degrees
        """
        return mesh.non_orthogonality_angle
    
    @staticmethod
    def calculate_skewness(mesh: vfm_mesh_geometry) -> torch.Tensor:
        """
        Calculate skewness for each face.
        
        Args:
            mesh: MeshGeometry object
            
        Returns:
            Tensor of shape [num_faces] containing skewness values
        """
        # Skewness is the ratio of the magnitude of the skewness vector
        # to the distance between cell centers
        skewness = torch.zeros(mesh.num_faces, device=mesh.device)
        
        for i in range(mesh.num_internal_faces):
            face_idx = mesh.internal_faces[i]
            skew_vector_mag = torch.norm(mesh.skewness_vectors[face_idx])
            cell_vector_mag = torch.norm(mesh.cell_center_vectors[face_idx])
            
            if cell_vector_mag > 0:
                skewness[face_idx] = skew_vector_mag / cell_vector_mag
        
        return skewness
    
    @staticmethod
    def report_quality_metrics(mesh: vfm_mesh_geometry) -> dict:
        """
        Report mesh quality metrics.
        
        Args:
            mesh: MeshGeometry object
            
        Returns:
            Dictionary containing quality metrics
        """
        non_ortho = MeshQuality.calculate_non_orthogonality(mesh)
        skewness = MeshQuality.calculate_skewness(mesh)
        
        # Calculate statistics for internal faces
        internal_non_ortho = non_ortho[mesh.internal_faces]
        internal_skewness = skewness[mesh.internal_faces]
        
        # Check if tensors are empty before calculating max/mean
        if len(internal_non_ortho) > 0 and len(internal_skewness) > 0:
            return {
                "max_non_orthogonality": float(torch.max(internal_non_ortho).item()),
                "avg_non_orthogonality": float(torch.mean(internal_non_ortho).item()),
                "max_skewness": float(torch.max(internal_skewness).item()),
                "avg_skewness": float(torch.mean(internal_skewness).item()),
            }
        else:
            return {
                "max_non_orthogonality": 0.0,
                "avg_non_orthogonality": 0.0,
                "max_skewness": 0.0,
                "avg_skewness": 0.0,
            }

