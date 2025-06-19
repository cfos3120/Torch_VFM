import torch
import numpy as np

def navier_stokes_2d(solution_field:torch.tensor, solution_index:dict, Re:torch.tensor, time_derivative:torch.tensor) -> torch.tensor:
    assert len(Re.shape) == len(solution_field.shape), f'{Re.shape} vs {solution_field.shape}'
    assert len(solution_field.shape) == 4, 'Solution field should be [batch, time-step, cells, channels]'

    p = solution_field[...,solution_index['p']]
    U = solution_field[...,solution_index['U']]
    assert U.shape[-1] == 2, 'Expecting a 2D field'

    # gradients in internal zone
    u_y  = (u[:, 2:  , 1:-1, 0] -   u[:,  :-2, 1:-1, 0]) / (2*dy)
    u_x  = (u[:, 1:-1, 2:  , 0] -   u[:, 1:-1,  :-2, 0]) / (2*dx)
    u_yy = (u[:, 2:  , 1:-1, 0] - 2*u[:, 1:-1, 1:-1, 0] + u[:,  :-2, 1:-1, 0]) / (dy**2)
    u_xx = (u[:, 1:-1, 2:  , 0] - 2*u[:, 1:-1, 1:-1, 0] + u[:, 1:-1,  :-2, 0]) / (dx**2)

    v_y  = (u[:, 2:  , 1:-1, 1] -   u[:,  :-2, 1:-1, 1]) / (2*dy)
    v_x  = (u[:, 1:-1, 2:  , 1] -   u[:, 1:-1,  :-2, 1]) / (2*dx)
    v_yy = (u[:, 2:  , 1:-1, 1] - 2*u[:, 1:-1, 1:-1, 1] + u[:,  :-2, 1:-1, 1]) / (dy**2)
    v_xx = (u[:, 1:-1, 2:  , 1] - 2*u[:, 1:-1, 1:-1, 1] + u[:, 1:-1,  :-2, 1]) / (dx**2)

    p_y  = (u[:, 2:  , 1:-1, 2] - u[:,  :-2, 1:-1, 2]) / (2*dy)
    p_x  = (u[:, 1:-1, 2:  , 2] - u[:, 1:-1,  :-2, 2]) / (2*dx)

    # Equations
    dict_keys = ['Continuity',
                 'X-momentum',
                 'Y-momentum']
    
    f0 = (u_x + v_y)
    f1 = u[:,1:-1,1:-1, 0]*u_x + u[:,1:-1,1:-1, 1]*u_y - (1/Re.reshape(B,1,1)) * (u_xx + u_yy) + p_x
    f2 = u[:,1:-1,1:-1, 0]*v_x + u[:,1:-1,1:-1, 1]*v_y - (1/Re.reshape(B,1,1)) * (v_xx + v_yy) + p_y

    equations = time_derivative + advection_field -(1/Re)*laplacian_field + pressure_gradient
    equations['Continuity'] = f0
    equations['X-momentum'] = f1
    equations['Y-momentum'] = f2