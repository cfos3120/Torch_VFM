from .navier_stokes_fdm import navier_stokes_2d_cavity
from .navier_stokes_fvm import navier_stokes_fvm
from .navier_stokes_autograd import navier_stokes_2d_cavity_auto

def pde_selector(method:str):
    method_list = {'navier_stokes_2d_cavity':navier_stokes_2d_cavity,
                   'navier_stokes_fvm':navier_stokes_fvm,
                   'navier_stokes_2d_cavity_auto':navier_stokes_2d_cavity_auto}
    return method_list[method]


def gradient_str(channel, mesh_dim:int=2, order:int=1, time_dim:bool=False):
    if mesh_dim == 2 and order == 1:
        mesh_den_str = ['dx','dy']
    elif mesh_dim == 2 and order == 2:
        mesh_den_str = ['dxx','dxy','dyx','dyy']
    elif mesh_dim == 3 and order == 1:
        mesh_den_str = ['dx','dy','dz']
    elif mesh_dim == 3 and order == 2:
        mesh_den_str = ['dxx','dxy','dxz','dyx','dyy','dyz','dzx','dzy','dzz']
    else:
        raise NotImplementedError(f'mesh_dim={mesh_dim} and order={order} not supported')
    
    if channel == 'U':
        if mesh_dim == 2:
            mesh_num_str = ['du','dv']
        elif mesh_dim == 3:
            mesh_num_str = ['du','dv','dw']
    else:
        mesh_num_str = [f'd{channel}']

    # override for time_dim
    if time_dim:
        mesh_den_str = ['dt']

    mesh_str = [[f'{num}/{den}'for den in mesh_den_str] for num in mesh_num_str]
    mesh_str = [item for sublist in mesh_str for item in sublist]
    return mesh_str