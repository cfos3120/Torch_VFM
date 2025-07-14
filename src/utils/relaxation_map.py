import torch
import numpy as np

'''
Purpose of function is to return an adjusted SDF that relaxes the intensity of a field (designed for PDE loss fields)
to help Physics Informed Machine Learning. It takes problem points (in this case hard coded through a separate call function),
as well as option boundary points (WIP) and produces either a Power Weighted SDF with impact radius equivalent to cell 
discretization (strictly specific for isometric meshes, loosely specific for complex meshes, but the same global variable dx)
times by the cell_multiples where any cell beyond dx*cell_multiples distance from the problem points is not relaxed.
'''

def get_problem_points(name):
    if name == 'cavity':
        return torch.tensor(np.stack([[0.0,1.0],[1.0,1.0]]))
    else:
        raise NotImplementedError(f'SDF relaxation mapping for {name} not implemented')

def get_relaxation_map(name, cells, dx=None, boundary_points=None, power=1, dims=[0,1], cell_multiples=1):
    problem_points = get_problem_points(name)
    return relaxation_sdf_map(cells, 
                              problem_points, 
                              dx=dx, 
                              boundary_points=boundary_points, 
                              power=power, 
                              dims=dims, 
                              cell_multiples=cell_multiples)


def relaxation_sdf_map(cells, 
                       problem_points, 
                       dx=None, 
                       boundary_points=None, 
                       power=1, 
                       dims=[0,1], 
                       cell_multiples=1):
    assert len(problem_points.shape) == 2
    assert problem_points.shape[-1] >= 2

    if boundary_points is not None:
        raise NotImplementedError("boundary_points support not mapped or calibrated")
    
    if dx is None:
        node_index = np.argmax(cells.shape)
        nodes = cells.shape[node_index]
        # assume square mesh, dx=dy
        resolution = int(np.sqrt(nodes)) 
        dx = 1/(resolution)

    deltas = cells[...,dims].unsqueeze(1) - problem_points.unsqueeze(0)
    dists = torch.norm(deltas, dim=2)  # (N, M)

    # SDF: minimum distance to any problem cell for each cell
    sdf = dists.min(dim=1).values  # (N,)
    sdf= torch.clamp(sdf, max=cell_multiples*dx)/(cell_multiples*dx)
    sdf_np = sdf.reshape(resolution, resolution).numpy()

    adjusted_sdf = sdf_np**power
    return adjusted_sdf
