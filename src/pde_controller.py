import torch
import numpy
from .gauss_green import *
from .utils.mesh_utils import get_vtk_file_reader

class pde_controller():
    def __init__(self, config: dict) -> None:
        config = self.config
        file_name = config['mesh_file_pointer']
        vtk_file_reader_fn = get_vtk_file_reader(file_name)
        vtk_file_reader = vtk_file_reader_fn(file_name)
        self.mesh = gaus_green_vfm_mesh(vtk_file_reader)

    def compute_losses():
        raise NotImplementedError 
    
    def balance_losses():
        raise NotImplementedError
    
    def compute(out: torch.tensor, y: torch.tensor = None) -> torch.tensor:
        '''
        If y is given, we assume that the time derivative for the first output time-step, should be 
        calculated using the last (or second last if we are using initial condition matching). Additional
        conditions to be set in __init__ are:
            1. Non-time dependent solutions
            2. Continuity only (for cases where dt is too large)
            3. Soblev Norms (need y for this) compare gradients as well and add to balancer

        NOTE: With FVM, we no longer need to add boundary coordinates to the field, as these are handled
        internally.
        '''



        raise NotImplementedErrorv
