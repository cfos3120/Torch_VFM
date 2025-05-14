import torch
import numpy as np

def fixed_value_bc(grad_field: torch.tensor,
                   field_values: torch.tensor,
                   owner: torch.tensor,
                   bc_patch: dict,
                   order = 1,
                   dtype = torch.float32,
                   d_vec:torch.tensor = None,
                   **kwargs) -> torch.tensor:
    '''
    Returns Tensor of shape like ,
    With the flux calculation of each boundary face.
    '''

    batch_size = grad_field.shape[0]
    channels = grad_field.shape[-1]
    time_dim = grad_field.shape[1]
    n_faces = len(owner)
    device = owner.device

    bc_value = np.array(bc_patch["value"])

    # (Batch, Time, Faces, Channels)
    bc_values_at_face = torch.tensor(bc_value, 
                                     dtype=dtype, 
                                     device=device
                                    ).reshape(1,1,1,-1).repeat(batch_size,time_dim,n_faces,1)
    if order == 2:
        return
        raise NotImplementedError
        # need to find the unit vector of the perpendicular distance from cell to face
        d_vec = torch.tensor(self.cell_coords, dtype = self.face_normals.dtype)[owner_patch] - self.face_centres[face_keys_idx]
        delta_d_vector = torch.einsum('ij,ij->i', d_vec, face_normals_patch).unsqueeze(-1) * face_normals_patch
        delta_d_unit_vector = torch.nn.functional.normalize(delta_d_vector, p=2, dim=1)
        
        # These cell values need to be changed to original U not GRAD(U)
        # The direction here may be wrong of the delta_d (should be pointing out of the cell, but difference is cell - face, might need to swap that)
        cell_values = field_values[...,owner,:] 
        bc_values_at_face = (cell_values - bc_values_at_face)
        #bc_values_at_face = torch.einsum('itjk,fc->btfci', bc_values_at_face, delta_d_unit_vector).permute(0,1,3,2).flatten(start_dim=2) #<- stuck here
        # we need Grad(U) at face = Delta(U)/Delta(D) keeping dimension and then flatten so that we have 
        # ['du/dx', 'du/dy', 'du/dz','dv/dx', 'dv/dy', 'dv/dz', 'dw/dx', 'dw/dy', 'dw/dz'] in the last dim channel, to match with our gradient field input.

        
        print(f'...added flux for {patch} (fV)')

    return bc_values_at_face

