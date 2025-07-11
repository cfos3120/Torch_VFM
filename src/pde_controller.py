import torch
from .gauss_green import *
from .utils.data_utils import get_vtk_file_reader, get_bc_dict
from .utils.training_utils import get_loss_fn
#from .physics.navier_stokes_basic import navier_stokes_2d
from .physics.utils import gradient_str
from .physics.navier_stokes_fvm import *
from .physics.navier_stokes_fdm import *
from .physics.utils import pde_selector

class pde_controller():
    def __init__(self, config: dict, device) -> None:
        self.verbose = True
        self.config = config
        file_name = config['mesh_file_pointer']
        self.vtk_file_reader = get_vtk_file_reader(file_name)
        self.device = device
        self.dim = 2
        self.mesh = gaus_green_vfm_mesh(self.vtk_file_reader, 
                                        L=config['length_scale'], 
                                        device=device, 
                                        bc_dict=get_bc_dict(type=config['bc_dict_type'])
                                        )
        
        self.input_f_indices = config['branch_key_index']

        self.pde_equations          = pde_selector(config['pde_equation'])
        self.verbose                = config['settings']['verbose']
        self.pin_first_ts           = config['settings']['pin_first_ts']
        self.ic_loss                = config['settings']['ic_loss']
        self.pde_loss               = config['settings']['pde_loss']
        self.mom_eqn_skip_first_ts  = config['settings']['mom_eqn_skip_first_ts']
        self.soblov_norms           = config['settings']['soblov_norms']
        self.dt_scheme              = config['settings']['dt_scheme']

        # indepenent control boolians:
        self.Re = 150
        self.time_step = 0.05
        self.boundary_nodes_set = False
        self.weight_pde = config['Re_batch_weight']
        
        self.loss_fn = get_loss_fn(config['loss_function'])

        self.epoch_loss_dict = {}
        
        self.field_channel_idx_dict = config['dataset_channels']
        self.enforcement_list = config['enforcement_list']

        if 'equations_limiters' in config.keys():
            self.limiters_dict = config['equations_limiters']
        else:
            self.limiters_dict = None
        if self.enforcement_list:
            assert set(self.enforcement_list).issubset(set(self.get_available_losses()))

    # def to(self,device):
    #     self.device = device
    #     self.mesh.to(device)
    def change_pde_method(self, method:str):
        self.pde_equations = pde_selector(method)

    def get_available_losses(self):
        available_losses = ['X-momentum Loss', 'Y-momentum Loss' ,'Continuity Loss', 'IC Loss']
        if self.dim == 3:
            available_losses += ['Z-momentum Loss']
        return available_losses

    def balance_losses(self, method='mean'):
        if method == 'mean':
            total_loss = torch.stack([self.loss_dict[key] for key in self.enforcement_list]).sum()/len(self.loss_dict.keys())
            return total_loss
        elif method == None:
            return {key:self.loss_dict[key] for key in self.enforcement_list}
        else:
            raise NotImplementedError
    
    def compute(self, 
                out: torch.tensor, 
                y: torch.tensor = None, 
                input_functions:tuple=None,
                input_solution: torch.tensor = None, 
                time_step=None, 
                Re:torch.tensor=None,
                model_input_coords:torch.tensor=None) -> torch.tensor:
        
        if input_functions is not None:
            if 'input_solution' in self.input_f_indices.keys():
                input_solution = input_functions[self.input_f_indices['input_solution']]
            if 'Re' in self.input_f_indices.keys():
                Re = input_functions[self.input_f_indices['Re']]

        # Introduce time-dimension for tensor compatability
        out = out.unsqueeze(1) if len(out.shape) == 3 else out
        y = y.unsqueeze(1) if len(y.shape) == 3 else y
        if input_solution is not None:
            input_solution = input_solution.unsqueeze(1) if len(input_solution.shape) == 3 else input_solution
        
        if time_step is None:
            time_step = self.time_step
        
        # Each function below updates this dict with loss tensors
        self.loss_dict = {}
        self.device = out.device

        # Prepare Gradients
        if self.soblov_norms and y is not None:
            if self.verbose: print('...Attempting to calculate real gradients', flush=True)
            raise NotImplementedError
            real_grads = self.get_spatial_gradients(y)

        # Initial Condition Loss
        if self.ic_loss and input_solution is not None:
            if self.verbose: print('...Attempting to calculate IC conditions', flush=True)
            self.compute_ic_loss(out,input_solution,y)

        # PDE Loss
        if self.pde_loss:
            assert out.shape[-2] == self.mesh.mesh.n_cells, f'expected {self.mesh.mesh.n_cells} but got {out.shape[-2]}'
            if Re is None and self.Re is not None: 
                Re = torch.tensor(self.Re, dtype=torch.float32, device=out.device).reshape(1,1,1,1) # (batch,time,cells,channels)

            if self.verbose: print('...Attempting to calculate PDE', flush=True)
            self.compute_pde_loss(out=out, 
                                  input_solution=input_solution, 
                                  time_step=time_step, 
                                  Re=Re, 
                                  model_input_coords=model_input_coords)

        if self.limiters_dict is not None:
            self.apply_limiters()
        if not self.enforcement_list:
            return None
        else:
            return self.balance_losses(method=None)

    def compute_ic_loss(self, out:torch.tensor,input_solution=None,y=None):
        if input_solution is not None and self.pin_first_ts:
            assert out.shape[-2] == input_solution.shape[-2]
            ic_loss = self.loss_fn(out[:,0,...], input_solution[:,-1,...])
        elif y is not None:
            ic_loss = self.loss_fn(out[:,0,...], y[:,0,...])
        else:
            raise ValueError('Either an input_solution or ground truth as inputs')
        
        self.loss_dict['IC Loss'] = ic_loss
        return ic_loss

    def compute_pde_field(self, 
                         out:torch.tensor,
                         input_solution:torch.tensor=None, 
                         time_step=None, 
                         Re:torch.tensor=None,
                         model_input_coords:torch.tensor=None):
        
        if len(out.shape) == 3 and self.dt_scheme == 'steady':
            out = out.unsqueeze(1) # add time dim

        if input_solution is not None:
            if self.pin_first_ts is False or input_solution.shape[1] == 1:
                input_solution = input_solution[...,self.field_channel_idx_dict['U']]
            else:
                input_solution = input_solution[:,:-1,:,self.field_channel_idx_dict['U']]
        dt_field = Temporal_Differentiator.caclulate(out[...,self.field_channel_idx_dict['U']], 
                                                     time_step=time_step, 
                                                     input_solution=input_solution, 
                                                     method=self.dt_scheme)
        # If we need gradients for Soblov loss, lets move this outside the function
        eqn_dict = self.pde_equations(mesh=self.mesh, 
                                    solution_field=out, 
                                    solution_index=self.field_channel_idx_dict, 
                                    Re=Re.unsqueeze(1), 
                                    time_derivative=dt_field,
                                    model_input_coords=model_input_coords)
        
        return eqn_dict

    def compute_pde_loss(self, 
                         out:torch.tensor, 
                         input_solution:torch.tensor=None, 
                         time_step=None, 
                         Re:torch.tensor=None,
                         model_input_coords:torch.tensor=None):
        
        eqn_dict = self.compute_pde_field(out=out,
                                          input_solution=input_solution,
                                          time_step=time_step,
                                          Re=Re,
                                          model_input_coords=model_input_coords)
        
        eqn_loss_dict = dict.fromkeys(eqn_dict.keys())
        for key in eqn_loss_dict:
            if self.weight_pde:
                eqn_loss_dict[key] = self.loss_fn(eqn_dict[key]*Re.reshape(-1,1,1), torch.zeros_like(eqn_dict[key]))
            else:
                eqn_loss_dict[key] = self.loss_fn(eqn_dict[key], torch.zeros_like(eqn_dict[key]))
            self.loss_dict[f'{key} Loss'] = eqn_loss_dict[key]

        return eqn_loss_dict
    
    def apply_limiters(self) -> None:
        '''
        In a traditional CFD solver, we specify a residual limit for the the equations to be solved to.
        In ML training, a pure zero residual may resemble a null field. To avoid this we cap the losses
        to a specified 'residual'. Additionally, we may want to reduce below that residual, under the
        condition that all equations have met their residual.
        '''
        for key, value in self.limiters_dict.items():
            if key in self.enforcement_list:
                self.loss_dict[key] = torch.max(self.loss_dict[key], torch.tensor(value,dtype=torch.float32,device=self.device))
        return

    def wandb_logging(self) -> dict:
        # if in the wandb logging we add a prefix such as 'weights/loss_func'
        # it will automatically group it in the report making it easier
        wandb_dict = {f'pde_controller/{key}': value.item() for key, value in self.loss_dict.items()}
        return wandb_dict
    
    def include_boundary_idx(self,idx):
        self.boundary_nodes_set = True
        self.boundary_node_strt_idx = idx

    def exclude_boundary_nodes(self,input):
        if not self.boundary_nodes_set:
            return input
    
        if input.shape[-1] > self.boundary_node_strt_idx:
            # Nodes are last channel (e.g. PDE equations)
            # Unlikely we will have more output channels than nodes
            input = input[...,:self.boundary_node_strt_idx]
        elif input.shape[-2] > self.boundary_node_strt_idx:
            # If second from last dim is nodes: expected
            input = input[...,:self.boundary_node_strt_idx,:]
        else:
            raise KeyError(f'{self.boundary_node_strt_idx}, is not greater than either dim -1,-2')
        return input