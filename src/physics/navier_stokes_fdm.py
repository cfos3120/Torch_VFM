import torch
import numpy as np

def navier_stokes_2d(solution_field:torch.tensor, solution_index:dict, Re:torch.tensor, time_derivative:torch.tensor=None) -> torch.tensor:
    assert len(Re.shape) == len(solution_field.shape), f'{Re.shape} vs {solution_field.shape}'
    assert len(solution_field.shape) == 4, 'Solution field should be [batch, time-step, cells, channels]'

    B, T, N, C = solution_field.shape

    # Unflatten and pad with ghost nodes (assuming Isometric grid)
    H = W = int(N**0.5)
    assert H * W == N, "N must be a perfect square"
    solution_field = solution_field.view(B, T, H, W, C)
    solution_field = torch.nn.functional.pad(solution_field, (0,0,1,1,1,1))
    
    # assume dx based on unit length domain (specific to cavity case)
    dx = dy = 1/(H-1)

    # split into components
    p = solution_field[...,solution_index['p']]
    U = solution_field[...,solution_index['U']]
    assert U.shape[-1] == 2, 'Expecting a 2D field'

    # Fill out ghost nodes based on boundary conditions (hard coded)
    # Lid
    U[:,-1  , :  , 0] = -U[:,-2  , :  , 0] + 2.0
    U[:,-1  , :  , 1] = -U[:,-2  , :  , 1] 
    p[:,-1  , :  , 0] =  p[:,-2  , :  , 0]

    # Left Wall
    U[:, :  , 0  , :2] = -U[:, :  , 1  , :2]
    p[:, :  , 0  , 0]  =  p[:, :  , 1  , 0]

    # Bottom Wall
    U[:, 0  , :  , :2] = -U[:, 1  , :  , :2]
    p[:, 0  , :  , 0] =  p[:, 1  , :  , 0]

    # Right Wall
    U[:, :  ,-1  , :2] = -U[:, :  ,-2  , :2]
    p[:, :  ,-1  , 0] =   p[:, :  ,-2  , 0]

    # gradients in internal zone
    u_y  = (U[:, 2:  , 1:-1, 0] -   U[:,  :-2, 1:-1, 0]) / (2*dy)
    u_x  = (U[:, 1:-1, 2:  , 0] -   U[:, 1:-1,  :-2, 0]) / (2*dx)
    u_yy = (U[:, 2:  , 1:-1, 0] - 2*U[:, 1:-1, 1:-1, 0] + U[:,  :-2, 1:-1, 0]) / (dy**2)
    u_xx = (U[:, 1:-1, 2:  , 0] - 2*U[:, 1:-1, 1:-1, 0] + U[:, 1:-1,  :-2, 0]) / (dx**2)

    v_y  = (U[:, 2:  , 1:-1, 1] -   U[:,  :-2, 1:-1, 1]) / (2*dy)
    v_x  = (U[:, 1:-1, 2:  , 1] -   U[:, 1:-1,  :-2, 1]) / (2*dx)
    v_yy = (U[:, 2:  , 1:-1, 1] - 2*U[:, 1:-1, 1:-1, 1] + U[:,  :-2, 1:-1, 1]) / (dy**2)
    v_xx = (U[:, 1:-1, 2:  , 1] - 2*U[:, 1:-1, 1:-1, 1] + U[:, 1:-1,  :-2, 1]) / (dx**2)

    p_y  = (p[:, 2:  , 1:-1, 0] - p[:,  :-2, 1:-1, 0]) / (2*dy)
    p_x  = (p[:, 1:-1, 2:  , 0] - p[:, 1:-1,  :-2, 0]) / (2*dx)

    # Equations
    f0 = u_x + v_y
    f1 = U[:,1:-1,1:-1, 0]*u_x + U[:,1:-1,1:-1, 1]*u_y - (1/Re) * (u_xx + u_yy) + p_x
    f2 = U[:,1:-1,1:-1, 0]*v_x + U[:,1:-1,1:-1, 1]*v_y - (1/Re) * (v_xx + v_yy) + p_y

    f0 = f0.view(B, T, H * W)
    f1 = f1.view(B, T, H * W)
    f2 = f2.view(B, T, H * W)

    equations = {'Continuity':f0, 'X-momentum':f1, 'Y-momentum':f2}