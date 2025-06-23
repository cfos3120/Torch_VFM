import torch
import vtk
import numpy as np
import pyvista as pv
from .utils.mesh_utils import *
from .vfm_mesh import *
from .utils.data_utils import get_bc_dict
    
class gaus_green_vfm_mesh():
    def __init__(self, vtk_file_reader, L=1, device='cpu', dtype=torch.float32, bc_dict:dict=None) -> None:

        self.device = device
        self.dtype = dtype
        
        # Read in Mesh in VTK
        vtk_file_reader.set_active_time_value(vtk_file_reader.time_values[-1])
        vtk_file_reader.cell_to_point_creation = False
        vtk_file_reader.enable_all_patch_arrays()
        self.vtk_mesh = vtk_file_reader.read()[0].scale(1/L)
    
        self.mesh = vfm_mesh_geometry(self.vtk_mesh, device=device, dtype=dtype)
        print(MeshQuality.report_quality_metrics(self.mesh))

        # Find boundary indices:
        #try:
        # TODO: replace this with VTK Boundary finder
        if bc_dict is not None:
            self.boundaries = vtk_file_reader.read()['boundary']
            for key in self.boundaries.keys():
                self.boundaries[key] = self.boundaries[key].scale(1/L)

            self.add_bc_conditions(bc_dict)
        else:
            self.boundaries = None
            print('No Boundary Patches Found')
        
        # Try adding boundaries:
        self.mesh._add_boundaries(self.boundaries)

        # Default settings:
        self.limiting_coefficient = 0.0
        self.correction_method = 'Over-Relaxed'
        self.implicit_orthogonality = True

        # Prepare interpolation weights
        self._calculate_correction_vectors(method = self.correction_method)
        self._calculate_internal_interpolation_weights()

    def _calculate_internal_interpolation_weights(self) -> torch.Tensor:
        
        self.internal_face_weights = torch.zeros(self.mesh.num_internal_faces, 
                                                device=self.device,
                                                dtype=self.dtype)

        for i, face_key in enumerate(self.mesh.internal_faces):
            p1, p2, p3, p4 = self.mesh.vertices[self.mesh.faces[face_key]]
            A, B, C, D = plane_equation_from_points(p1.cpu(), p2.cpu(), p3.cpu())

            # Line direction vector
            owner_cell_coord = self.mesh.cell_centers[self.mesh.face_owners[face_key]]
            neighbour_cell_coord = self.mesh.cell_centers[self.mesh.face_neighbors[face_key]]
            line_dir = self.mesh.cell_center_vectors[face_key]

            denominator = A * line_dir[0] + B * line_dir[1] + C * line_dir[2]
            t = -(A * owner_cell_coord[0] + B * owner_cell_coord[1] + C * owner_cell_coord[2] + D) / denominator

            intersect_coord = owner_cell_coord + t * line_dir
            self.internal_face_weights[i] = torch.norm(intersect_coord - neighbour_cell_coord, dim=-1, keepdim=False) / torch.norm(owner_cell_coord - neighbour_cell_coord, dim=-1, keepdim=False)

        print(f'Calculating Cell2Cell at Face Linear Interpolation Weights (L2):\n  min w:{self.internal_face_weights.min():.4f}, \
              max w:{self.internal_face_weights.max():.4f}, \
              mean w:{self.internal_face_weights.mean():.4f}')
        
    def _calculate_correction_vectors(self, method=None) -> torch.Tensor:
        assert method in ['Minimum','Orthogonal','Over-Relaxed', None], "Method must be one of 'Minimum', 'Orthogonal', or 'Over-Relaxed'."

        # initialize vectors
        self.d = torch.zeros(self.mesh.num_faces,
                        self.mesh.dim, 
                        device=self.device,
                        dtype=self.dtype)
        
        # fill in boundary faces with distance between cell center and face center
        self.d[self.mesh.internal_faces] = self.mesh.cell_center_vectors[self.mesh.internal_faces]
        self.d[self.mesh.boundary_faces] = self.mesh.face_centers[self.mesh.boundary_faces] - self.mesh.cell_centers[self.mesh.face_owners[self.mesh.boundary_faces]]
        self.d_mag = torch.norm(self.d, dim=-1, keepdim=False)

        if method == 'Minimum':
            # Minimum distance correction
            self.delta = self.mesh.face_normal_unit_vectors*(torch.einsum('fd,fd->f',self.mesh.face_normal_unit_vectors, self.d)/self.d_mag).unsqueeze(-1)
        elif method == 'Orthogonal':
            self.delta = self.d/self.d_mag.unsqueeze(-1)
        elif method == 'Over-Relaxed':
            self.delta = self.d * (1/torch.max(torch.einsum('fd,fd->f',self.mesh.face_normal_unit_vectors,self.d),0.5*self.d_mag)).unsqueeze(-1)
        else:
            self.delta = self.mesh.face_normal_unit_vectors
        
        self.delta_mag = torch.norm(self.delta, dim=-1, keepdim=False)

        # Calculate k_vector
        self.k_vector = self.mesh.face_normal_unit_vectors - self.delta
        self.k_vector_mag = torch.norm(self.k_vector, dim=-1, keepdim=False)
    
    def add_bc_conditions(self, dict):
        # TODO: this can to be replaced by something more automated or in init file
        filtered_dict = {
            field: {
                patch: settings
                for patch, settings in field_dict.items()
                if settings.get("type") != "empty"
            }
            for field, field_dict in dict.items()
        }
        
        # Get all valid patch names that remained in *any* field
        valid_patches = set().union(*[field_dict.keys() for field_dict in filtered_dict.values()])

        filtered_second_dict = {
            key: self.boundaries[key]
            for key in self.boundaries.keys()
            if key in valid_patches
        }
        
        self.boundaries = filtered_second_dict
        self.bc_conditions = filtered_dict

if __name__ == '__main__':
    print('hello world')

    dir = r'C:\Users\Noahc\Downloads\c5_test\case.foam'
    vtk_file_reader = pv.POpenFOAMReader(dir)
    mesh = gaus_green_vfm_mesh(vtk_file_reader, device='cpu')

    mesh.add_bc_conditions(get_bc_dict())



    # Sample Solution
    vtk_file_reader.set_active_time_value(vtk_file_reader.time_values[-1])
    vtk_file_reader.cell_to_point_creation = False
    vtk_file_reader.enable_all_patch_arrays()
    vtk_mesh = vtk_file_reader.read()[0]

    U_gt = torch.tensor(vtk_mesh['U'], dtype = torch.float32)
    gradU_gt = torch.tensor(vtk_mesh['grad(U)'], dtype = torch.float32)
    lapU_gt = torch.tensor(vtk_mesh['lapU'], dtype = torch.float32)
    divU_gt = torch.tensor(vtk_mesh['divU'], dtype = torch.float32)

    # Get values
    gradU = mesh.calculate_gradients(U_gt.unsqueeze(0).unsqueeze(0), field_type='U')
    print(f'Any NA in gradU: {torch.any(torch.isnan(gradU))}')

    divU = mesh.calculate_divergence(U_gt.unsqueeze(0).unsqueeze(0), field_type='U')
    print(f'Any NA in divU: {torch.any(torch.isnan(divU))}')

    for i in ['Minimum', 'Orthogonal', 'Over-Relaxed', None]:
        for j in [True, False]:
            mesh.limiting_coefficient = 0.0
            mesh.correction_method = i
            mesh.implicit_orthogonality = j
            lapU = mesh.calculate_laplacian(U_gt.unsqueeze(0).unsqueeze(0), field_type='U')
            print(f'mesh.correction_method{mesh.correction_method}, mesh.implicit_orthogonality {mesh.implicit_orthogonality} success',
                  f' Any NA in gradU: {torch.any(torch.isnan(lapU))}')