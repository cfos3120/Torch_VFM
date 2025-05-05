import torch
import numpy as np

def symmetry_plane_bc(field_values: torch.tensor, 
                   owner: torch.tensor,
                   face_normals_patch: torch.tensor,
                   order = 1,
                   **kwargs) -> torch.tensor:
    
    batch_size = field_values.shape[0]
    channels = field_values.shape[-1]
    time_dim = field_values.shape[1]
    n_faces = len(owner)
    device = owner.device

    if order > 1:
        return
    
    if channels > 1:
        cell_values = field_values[...,owner,:] 
        bc_values_at_face = cell_values - 2*(torch.einsum('btfc,fc->btf', cell_values, face_normals_patch)).unsqueeze(-1) * face_normals_patch.unsqueeze(0)
        return bc_values_at_face