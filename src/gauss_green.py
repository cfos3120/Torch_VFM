import torch
import vtk
import numpy as np
import pyvista as pv
#from .utils.mesh_utils import *
#from .vfm_mesh import *
#from .utils.data_utils import get_bc_dict

import sys
import os
sys.path.insert(0, r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\Torch_VFM')
from src.utils.mesh_utils import *
from src.vfm_mesh import *
from src.utils.data_utils import get_bc_dict

        
class gaus_green_vfm_mesh():
    def __init__(self, vtk_file_reader, L=1, device='cpu', dtype=torch.float32) -> None:

        self.device = device
        self.dtype = dtype
        
        # Read in Mesh in VTK
        vtk_file_reader.set_active_time_value(vtk_file_reader.time_values[-1])
        vtk_file_reader.cell_to_point_creation = False
        vtk_file_reader.enable_all_patch_arrays()
        self.vtk_mesh = vtk_file_reader.read()[0].scale(1/L)
    
        self.mesh = vfm_mesh_geometry(self.vtk_mesh, device, dtype=dtype)
        print(MeshQuality.report_quality_metrics(self.mesh))

        # Find boundary indices:
        try:
            # TODO: replace this with VTK Boundary finder
            self.boundaries = vtk_file_reader.read()['boundary']
            for key in self.boundaries.keys():
                self.boundaries[key] = self.boundaries[key].scale(1/L)
        except:
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
            A, B, C, D = plane_equation_from_points(p1, p2, p3)

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
        for key1 in dict: # U or P
            for key2 in dict[key1]: # patch
                if 'value' in dict[key1][key2].keys():
                    if isinstance(dict[key1][key2]['value'], list):
                        dict[key1][key2]['value'] = dict[key1][key2]['value'][:self.mesh.dim]
        self.bc_conditions = dict

    def apply_bc_conditions(self, patch_name, bc_face_idx, field, field_type:str='U',grad_field_f:bool=False, original_field:torch.Tensor=None) -> torch.Tensor:
        raise NotImplementedError
        patch_type = self.bc_conditions[field_type][patch_name]['type']
        bc_cell_idx = self.mesh.face_owners[bc_face_idx]
        batch_size = field.shape[0]
        time_size = field.shape[1]
        
        if patch_type == 'empty':
            return None
        elif patch_type == 'fixedValue':
            field_value = torch.tensor(self.bc_conditions[field_type][patch_name]['value'], dtype=self.dtype, device=self.device)
            if grad_field_f:
                assert original_field is not None
                delta_phi = original_field[...,bc_cell_idx,:] - field_value.reshape(1, 1, 1, -1).repeat(batch_size, time_size, len(bc_face_idx), 1)
                # repeated U so that it is U,U,U,V... etc and repeated delta so that it is dx,dy,dz,dx... etc.
                #face_grad = torch.einsum('btfd,fd->btfd', delta_phi.repeat_interleave(self.mesh.dim, dim=-1), (self.d/self.d_mag.unsqueeze(-1))[bc_face_idx,:])
                face_grad = torch.einsum('btfd,fe->btfed', delta_phi, (self.d/self.d_mag.unsqueeze(-1))[bc_face_idx,:]).flatten(start_dim=-2)
                
                return face_grad
            else:
                return field_value.reshape(1, 1, 1, -1).repeat(batch_size, time_size, len(bc_face_idx), 1)
        elif patch_type == 'zeroGradient':
            if grad_field_f:
                return None
            else:
                return field[...,bc_cell_idx,:]
        elif patch_type == 'noSlip':
            if grad_field_f:
                assert original_field is not None
                delta_phi = original_field[...,bc_cell_idx,:]
                face_grad = torch.einsum('btfd,fe->btfed', delta_phi, (self.d/self.d_mag.unsqueeze(-1))[bc_face_idx,:])
                face_grad_normal = torch.einsum('btfed,fd->btfe', face_grad, self.mesh.face_normal_unit_vectors[bc_face_idx,:])*self.mesh.face_normal_unit_vectors[bc_face_idx,:]
                return face_grad_normal.flatten(start_dim=-2)
            else:
                return None
        elif patch_type == 'symmetryPlane':
            if grad_field_f:
                return None
            else:
                return field[...,bc_cell_idx,:] - 2*(torch.einsum('btfc,fc->btf', field[...,bc_cell_idx,:], self.mesh.face_normal_unit_vectors[bc_face_idx,:])).unsqueeze(-1) * self.mesh.face_normal_unit_vectors[bc_face_idx,:].unsqueeze(0)

    def interpolate_to_faces(self, field: torch.Tensor, field_type:str='U', grad_field_f=False, original_field=None) -> torch.Tensor:
        raise NotImplementedError
        # Get field shape
        assert len(field.shape) == 4
        field_shape = field.shape
        batch_size = field_shape[0]
        time_size = field_shape[1]
        channel_size = field_shape[-1]
        
        # Initialize face values
        face_values = torch.zeros((batch_size, time_size, self.mesh.num_faces, channel_size), device=field.device)
        
        # Interpolate for internal faces
        idx = self.mesh.internal_faces
        face_values[:,:,idx,:] = field[:,:,self.mesh.face_owners[idx],...]*(self.internal_face_weights).reshape(1,1,-1,1)  + \
                                 field[:,:,self.mesh.face_neighbors[idx],...]*(1-self.internal_face_weights).reshape(1,1,-1,1)
        
        # Apply hard boundary conditions for faces
        if self.boundaries is not None:
            for patch_name, patch_faces in self.mesh.patch_face_keys.items():
                face_value = self.apply_bc_conditions(patch_name, 
                                                      bc_face_idx=patch_faces, 
                                                      field=field, 
                                                      field_type=field_type, 
                                                      grad_field_f=grad_field_f, 
                                                      original_field=original_field)
                if face_value is not None:
                    face_values[:,:,patch_faces,:] = face_value
        
        return face_values
    
    def calculate_gradients(self, field: torch.Tensor, field_type:str, face_values:torch.Tensor=None) -> torch.Tensor:
        raise NotImplementedError
        # Get field shape
        field_shape = field.shape
        batch_size = field_shape[0]
        time_size = field_shape[1]
        channel_size = field_shape[-1]

        # Interpolate field to face centers
        if face_values is None:
            face_values = self.interpolate_to_faces(field, field_type)
        
        # Calculate gradients using Gauss-Green theorem
        face_flux = torch.einsum('btfd,fe->btfed', face_values, self.mesh.face_areas)
        
        # Add to owner/neighbour cells
        idx = self.mesh.internal_faces
        gradients = torch.zeros((batch_size, time_size, self.mesh.n_cells, channel_size, self.mesh.dim), device=field.device)
        gradients.index_add_(2, self.mesh.face_owners, face_flux)
        gradients.index_add_(2, self.mesh.face_neighbors[idx], -face_flux[:,:,idx,:,:])
        gradients = gradients / self.mesh.cell_volumes.reshape(1,1,-1,1,1)
        return gradients.flatten(start_dim=-2)
    
    def calculate_divergence(self, field: torch.Tensor, field_type:str, grad_field=False) -> torch.Tensor:
        raise NotImplementedError
        # Get field shape
        field_shape = field.shape
        batch_size = field_shape[0]
        time_size = field_shape[1]
        channel_size = field_shape[-1]
        
        div_field = torch.zeros((batch_size, time_size, self.mesh.n_cells, self.mesh.dim), dtype=field.dtype, device=self.device)
        
        face_values = self.interpolate_to_faces(field, field_type, grad_field)
        face_flux_mag = torch.einsum('btfd,fd->btf', face_values, self.mesh.face_areas)
        divergence  = face_flux_mag.unsqueeze(-1) * face_values

        idx = self.mesh.internal_faces
        div_field = torch.zeros((batch_size, time_size, self.mesh.n_cells, self.mesh.dim), dtype=field.dtype, device=self.device)
        div_field.index_add_(2, self.mesh.face_owners, divergence)
        div_field.index_add_(2, self.mesh.face_neighbors[idx], -divergence[...,idx,:])
        div_field /= self.mesh.cell_volumes.reshape(1,1,-1,1)
        return div_field

    def calculate_laplacian(self, field: torch.Tensor, field_type:str, face_values:torch.Tensor=None) -> torch.Tensor:
        raise NotImplementedError
        # Get field shape
        field_shape = field.shape
        batch_size = field_shape[0]
        time_size = field_shape[1]
        channel_size = field_shape[-1]

        # initialize laplacian field
        lap_field = torch.zeros((batch_size, time_size, self.mesh.n_cells, self.mesh.dim), dtype=field.dtype, device=self.device)

        # Interpolate field to face centers
        if face_values is None:
            face_values = self.interpolate_to_faces(field, field_type)
        
        # Calculate Implicit Face Gradients using First Order Appoximation
        implicit_face_gradients = torch.zeros((batch_size, time_size, self.mesh.num_faces, channel_size), dtype=field.dtype, device=field.device)
        implicit_face_gradients[:,:,self.mesh.internal_faces,:] = field[:,:,self.mesh.face_neighbors[self.mesh.internal_faces], :] - field[:,:,self.mesh.face_owners[self.mesh.internal_faces],:]
        implicit_face_gradients[:,:,self.mesh.boundary_faces,:] = face_values[:,:,self.mesh.boundary_faces, :] - field[:,:,self.mesh.face_owners[self.mesh.boundary_faces],:]
        implicit_face_gradients /= self.d_mag.reshape(1, 1, -1, 1)

        # Calculate Explicit Face Gradients using Gauss-Green theorem
        gradients = self.calculate_gradients(field, field_type, face_values=face_values)
        explicit_face_gradients = self.interpolate_to_faces(gradients, field_type, grad_field_f=True, original_field=field)
        explicit_face_gradients = explicit_face_gradients.unflatten(dim=-1, sizes=(self.mesh.dim, self.mesh.dim))
            

        if self.implicit_orthogonality:
            # Implicit has an approximation for grad(U) dot Delta
            #orth_diffusion = torch.einsum('btfd,fd->btfd', implicit_face_gradients, self.delta_mag) * self.mesh.face_areas_mag.reshape(1, 1, -1, 1)
            orth_diffusion = implicit_face_gradients * (self.delta_mag * self.mesh.face_areas_mag).reshape(1, 1, -1, 1)
        else:
            # Explicit executes the dot_product exactly
            orth_diffusion = torch.einsum('btfde,fe->btfd', explicit_face_gradients, self.delta) * self.mesh.face_areas_mag.reshape(1, 1, -1, 1)
            
        if self.correction_method is not None:
            non_orth_diffusion = torch.einsum('btfde,fe->btfd', explicit_face_gradients, self.k_vector) * self.mesh.face_areas_mag.reshape(1, 1, -1, 1)
        else:
            non_orth_diffusion = 0

        face_diffusion = orth_diffusion + non_orth_diffusion

        # Add to owner/neighbour cells
        idx = self.mesh.internal_faces
        lap_field = torch.zeros((batch_size, time_size, self.mesh.n_cells, self.mesh.dim), dtype=field.dtype, device=self.device)
        lap_field.index_add_(2, self.mesh.face_owners, face_diffusion)
        lap_field.index_add_(2, self.mesh.face_neighbors[idx], -face_diffusion[...,idx,:])
        lap_field /= self.mesh.cell_volumes.reshape(1,1,-1,1)
        return lap_field
            


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