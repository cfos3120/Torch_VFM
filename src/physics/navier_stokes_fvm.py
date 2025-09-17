import torch
from .operators import *

def navier_stokes_fvm(mesh, solution_field:torch.tensor, solution_index:dict, Re:torch.tensor, time_derivative:torch.tensor, settings=None, **kwargs) -> torch.tensor:
    '''
    grad_dict should be a dictionary of tensors, with keys aligning to the following format:
    dp/dx for scalars and for vectors
    dU/dx are denoted as du/dx and dv/dy same with time derivates being du/dt and dv/dt
    '''
    

    assert len(Re.shape) == len(solution_field.shape), f'{Re.shape} vs {solution_field.shape}'
    assert len(solution_field.shape) == 4, 'Solution field should be [batch, time-step, cells, channels]'

    p = solution_field[...,solution_index['p']]
    U = solution_field[...,solution_index['U']]
    if U.shape[-1] == 2:
        flag2d = True
        U = torch.nn.functional.pad(U,(0,1))
    else:
        flag2d = False
    
    advection_field, gradient_field = Divergence_Operator.caclulate(mesh,U,field_type = 'U')
    _, pressure_gradient            = Divergence_Operator.caclulate(mesh,p,field_type = 'p')
    laplacian_field                 = Laplacian_Operator.caclulate(mesh,U,field_type = 'U', 
                                                                   gradient_field=gradient_field, 
                                                                   correction_method=mesh.correction_method)
    
    dudx = gradient_field[...,0]
    dvdy = gradient_field[...,4]
    dwdz = gradient_field[...,8]

    # Equations
    dict_keys = ['Continuity',
                 'X-momentum',
                 'Y-momentum']
    
    equations = dict.fromkeys(dict_keys)
    momentum_equations = time_derivative + advection_field -(1/Re)*laplacian_field + pressure_gradient
    equations['X-momentum'] = momentum_equations[...,0]
    equations['Y-momentum'] = momentum_equations[...,1]

    if flag2d:
        equations['Continuity'] = dudx + dvdy
    else: 
        equations['Continuity'] = dudx + dvdy + dwdz
        equations['Z-momentum'] = momentum_equations[...,2]

    if settings['volume_weighted']:
        if settings['volume_direct_scaling']:
            multiplier = mesh.mesh.cell_volumes.reshape(1,1,-1)*settings['total_multipier']
        else:
            multiplier = mesh.mesh.cell_volumes.reshape(1,1,-1)
        for key, values in equations.items():
            equations[key] = multiplier*values

    return equations