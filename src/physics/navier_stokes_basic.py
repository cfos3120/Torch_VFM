import torch

def navier_stokes_2d(solution_field:torch.tensor, solution_index:dict, Re:torch.tensor, grad_dict:dict) -> torch.tensor:
    '''
    grad_dict should be a dictionary of tensors, with keys aligning to the following format:
    dp/dx for scalars and for vectors
    dU/dx are denoted as du/dx and dv/dy same with time derivates being du/dt and dv/dt
    '''
    dict_keys = ['Continuity',
                 'X-momentum',
                 'Y-momentum']
    equations = dict.fromkeys(dict_keys)

    assert len(Re.shape) == len(solution_field.shape)-1, f'{Re.shape} vs {solution_field.shape}'

    U = solution_field[...,solution_index['U']]
    
    # Required variables
    u, v = U[...,0], U[...,1]

    # First Derivatives:
    dudx, dudy = grad_dict['du/dx'], grad_dict['du/dy']
    dvdx, dvdy = grad_dict['dv/dx'], grad_dict['dv/dy']
    dpdx, dpdy = grad_dict['dp/dx'], grad_dict['dp/dy']
    dudt, dvdt = grad_dict['du/dt'], grad_dict['dv/dt']
    
    # Second Derivatives:
    dudxx, dudyy = grad_dict['du/dxx'], grad_dict['du/dyy']
    dvdxx, dvdyy = grad_dict['dv/dxx'], grad_dict['dv/dyy']
    
    # Equations
    equations['Continuity'] = dudx + dvdy
    equations['X-momentum'] = dudt + u*dudx + v*dudy -(1/Re)*(dudxx+dudyy) + dpdx
    equations['Y-momentum'] = dvdt + u*dvdx + v*dvdy -(1/Re)*(dvdxx+dvdyy) + dpdy

    return equations
