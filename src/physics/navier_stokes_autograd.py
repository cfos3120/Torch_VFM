import torch

def navier_stokes_2d_cavity_auto(model_input_coords:torch.tensor, solution_field:torch.tensor, solution_index:dict, Re:torch.tensor, **kwargs):
    assert len(Re.shape) == len(solution_field.shape), f'{Re.shape} vs {solution_field.shape}'
    assert len(solution_field.shape) == 4, 'Solution field should be [batch, time-step, cells, channels]'
    assert solution_field.shape[1] == 1, 'function only handles steady cases'


    u = solution_field[...,solution_index['U'][0]].squeeze(1)
    v = solution_field[...,solution_index['U'][1]].squeeze(1)
    p = solution_field[...,solution_index['p']].squeeze(1)

    # First Derivatives
    u_out = torch.autograd.grad(u.sum(), model_input_coords, create_graph=True, retain_graph=True)[0]
    v_out = torch.autograd.grad(v.sum(), model_input_coords, create_graph=True, retain_graph=True)[0]
    p_out = torch.autograd.grad(p.sum(), model_input_coords, create_graph=True, retain_graph=True)[0]

    u_x = u_out[..., 0] #...might need to check the order of model_input_coords to ensure normal pressure boundary
    u_y = u_out[..., 1]

    v_x = v_out[..., 0]
    v_y = v_out[..., 1]

    p_x = p_out[..., 0]
    p_y = p_out[..., 1]
    
    # Second DerivativesTrue
    u_xx = torch.autograd.grad(u_x.sum(), model_input_coords, create_graph=False, retain_graph=True)[0][..., 0]
    u_yy = torch.autograd.grad(u_y.sum(), model_input_coords, create_graph=False, retain_graph=True)[0][..., 1]
    v_xx = torch.autograd.grad(v_x.sum(), model_input_coords, create_graph=False, retain_graph=True)[0][..., 0]
    v_yy = torch.autograd.grad(v_y.sum(), model_input_coords, create_graph=False, retain_graph=True)[0][..., 1]

    # Continuity equation
    f0 = u_x + v_y

    # Navier-Stokes equation
    f1 = u*u_x + v*u_y - (1/Re.reshape(-1,1)) * (u_xx + u_yy) + p_x
    f2 = u*v_x + v*v_y - (1/Re.reshape(-1,1)) * (v_xx + v_yy) + p_y

    return {'Continuity':f0.unsqueeze(1), 'X-momentum':f1.unsqueeze(1), 'Y-momentum':f2.unsqueeze(1)}