import torch
import numpy as np
from .fixed_value import *
from .symmetry_plane import *
from .zero_gradient import *

def get_function_map() -> dict:
    function_map = {'fixedValue': fixed_value_bc,
                    'zeroGradient': zero_gradient_bc,
                    'symmetryPlane': symmetry_plane_bc}
    
    return function_map
    
def get_boundary_flux(bc_patch,
                      grad_field,
                      field_values,
                      owner,
                      delta_d_vector,
                      original_field,
                      face_normals_patch,
                      order=1):

    '''
    bc_patch is dictionary with keys:
    - type
    - value (optional -> for fixedValue)
    '''
    function_map = get_function_map()
    
    boundary_function = function_map[bc_patch["type"]]

    bc_values_at_face = boundary_function(grad_field = grad_field, 
                                          field_values = field_values,
                                          bc_patch = bc_patch,
                                          owner = owner,
                                          order = order,
                                          delta_d_vector = delta_d_vector,
                                          original_field=original_field,
                                          face_normals_patch = face_normals_patch,
                                          dtype = torch.float64)
    return bc_values_at_face