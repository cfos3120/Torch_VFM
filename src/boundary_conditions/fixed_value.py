import torch
import numpy as np

def fixed_value_bc(grad_field: torch.tensor,
                   field_values: torch.tensor,
                   owner: torch.tensor,
                   bc_patch: dict,
                   order = 1,
                   dtype = torch.float32,
                   delta_d_vector:torch.tensor = None,
                   original_field:torch.tensor = None,
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
        # need to find the unit vector of the perpendicular distance from cell to face
        assert original_field is not None
        #return None
        # These cell values need to be changed to original U not GRAD(U)
        # The direction here may be wrong of the delta_d (should be pointing out of the cell, but difference is cell - face, might need to swap that)
        cell_values = original_field[...,owner,:]
        bc_values_at_face = (bc_values_at_face - cell_values)
        print(bc_values_at_face.shape, delta_d_vector.shape)
        delta_d_vector = delta_d_vector.unsqueeze(0).unsqueeze(0)
        disp_norm_sq = (delta_d_vector ** 2).sum(dim=-1, keepdim=True)
        jacobian = bc_values_at_face.unsqueeze(-1) * delta_d_vector.unsqueeze(-2)
        print(jacobian.shape, disp_norm_sq.shape)
        jacobian = jacobian / disp_norm_sq.unsqueeze(-1)
        bc_values_at_face = jacobian.flatten(start_dim=-2)
        
        #dmag = np.linalg.norm(delta_d_vector, axis=1, keepdims=True).reshape(1,1,-1,1,1)
        #bc_values_at_face = (torch.einsum('btfc,fl->btfcl', bc_values_at_face, delta_d_vector)/dmag).permute(0,1,2,4,3).flatten(start_dim=-2)
        #bc_values_at_face = torch.einsum('btfc,fl->btfcl', bc_values_at_face, 1/delta_d_vector).permute(0,1,2,4,3).flatten(start_dim=-2)

    return bc_values_at_face

