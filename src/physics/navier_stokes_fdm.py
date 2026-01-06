import torch
import numpy as np

def navier_stokes_2d_cavity(solution_field:torch.tensor, solution_index:dict, Re:torch.tensor, **kwargs) -> torch.tensor:
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
    U[...,-1  , :  , 0] = -U[...,-2  , :  , 0] + 2.0
    U[...,-1  , :  , 1] = -U[...,-2  , :  , 1] 
    p[...,-1  , :  , 0] =  p[...,-2  , :  , 0]

    # Left Wall
    U[..., :  , 0  , :2] = -U[..., :  , 1  , :2]
    p[..., :  , 0  , 0]  =  p[..., :  , 1  , 0]

    # Bottom Wall
    U[..., 0  , :  , :2] = -U[..., 1  , :  , :2]
    p[..., 0  , :  , 0] =  p[..., 1  , :  , 0]

    # Right Wall
    U[..., :  ,-1  , :2] = -U[..., :  ,-2  , :2]
    p[..., :  ,-1  , 0] =   p[..., :  ,-2  , 0]

    # gradients in internal zone
    u_y  = (U[..., 2:  , 1:-1, 0] -   U[...,  :-2, 1:-1, 0]) / (2*dy)
    u_x  = (U[..., 1:-1, 2:  , 0] -   U[..., 1:-1,  :-2, 0]) / (2*dx)
    u_yy = (U[..., 2:  , 1:-1, 0] - 2*U[..., 1:-1, 1:-1, 0] + U[...,  :-2, 1:-1, 0]) / (dy**2)
    u_xx = (U[..., 1:-1, 2:  , 0] - 2*U[..., 1:-1, 1:-1, 0] + U[..., 1:-1,  :-2, 0]) / (dx**2)

    v_y  = (U[..., 2:  , 1:-1, 1] -   U[...,  :-2, 1:-1, 1]) / (2*dy)
    v_x  = (U[..., 1:-1, 2:  , 1] -   U[..., 1:-1,  :-2, 1]) / (2*dx)
    v_yy = (U[..., 2:  , 1:-1, 1] - 2*U[..., 1:-1, 1:-1, 1] + U[...,  :-2, 1:-1, 1]) / (dy**2)
    v_xx = (U[..., 1:-1, 2:  , 1] - 2*U[..., 1:-1, 1:-1, 1] + U[..., 1:-1,  :-2, 1]) / (dx**2)

    p_y  = (p[..., 2:  , 1:-1, 0] - p[...,  :-2, 1:-1, 0]) / (2*dy)
    p_x  = (p[..., 1:-1, 2:  , 0] - p[..., 1:-1,  :-2, 0]) / (2*dx)

    # Equations
    f0 = u_x + v_y
    f1 = U[...,1:-1,1:-1, 0]*u_x + U[...,1:-1,1:-1, 1]*u_y - (1/Re.reshape(-1,1,1,1)) * (u_xx + u_yy) + p_x
    f2 = U[...,1:-1,1:-1, 0]*v_x + U[...,1:-1,1:-1, 1]*v_y - (1/Re.reshape(-1,1,1,1)) * (v_xx + v_yy) + p_y

    f0 = f0.view(B, T, H * W)
    f1 = f1.view(B, T, H * W)
    f2 = f2.view(B, T, H * W)

    return {'Continuity':f0, 'X-momentum':f1, 'Y-momentum':f2}

def periodic_derivatives(x, dx=1.0, dy=1.0):
    """
    Compute first and second derivatives with 2nd-order central differences
    and periodic BCs.
    
    x: tensor of shape (B, N, N) or (B, N, N, C)
    dx, dy: grid spacing
    
    Returns:
        dx1, dy1 : first derivatives
        dx2, dy2 : second derivatives
    """

    # shift along x-direction (dim=1)
    x_ip = torch.roll(x, shifts=-1, dims=1)
    x_im = torch.roll(x, shifts=+1, dims=1)

    # shift along y-direction (dim=2)
    y_ip = torch.roll(x, shifts=-1, dims=2)
    y_im = torch.roll(x, shifts=+1, dims=2)

    # First derivatives: (f[i+1] - f[i-1]) / (2*dx)
    dx1 = (x_ip - x_im) / (2.0 * dx)
    dy1 = (y_ip - y_im) / (2.0 * dy)

    # Second derivatives: (f[i+1] - 2f[i] + f[i-1]) / dx^2
    dx2 = (x_ip - 2.0 * x + x_im) / (dx * dx)
    dy2 = (y_ip - 2.0 * x + y_im) / (dy * dy)

    return dx1, dy1, dx2, dy2