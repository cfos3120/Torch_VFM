import torch
from .gauss_green import *
from .utils.data_utils import get_vtk_file_reader, get_bc_dict
from .physics.navier_stokes_basic import navier_stokes_2d
from .physics.utils import gradient_str

class pde_controller():
    def __init__(self, config: dict) -> None:
        self.verbose = True
        self.config = config
        file_name = config['mesh_file_pointer']
        self.vtk_file_reader = get_vtk_file_reader(file_name)
        self.mesh = gaus_green_vfm_mesh(self.vtk_file_reader)

        # prepare mesh
        self.mesh.add_bc_conditions(get_bc_dict())
        self.mesh.patch_face_keys_dict(exclusion_list=['front', 'back'])

        self.pde_equations = navier_stokes_2d # <- we can make this a map for other pdes
        self.verbose = True
        self.ic_loss = True
        self.pde_loss = True
        self.soblov_norms = False

        # indepenent control boolians:
        self.mom_eqn_skip_first_ts = False
        self.Re = 150
        self.time_step = 0.05
        
        self.loss_fn = torch.nn.MSELoss()

        self.epoch_loss_dict = {}
        
        self.field_channel_idx_dict = config['dataset_channels']
        self.enforcement_list = config['enforcement_list']
        if 'equations_limiters' in config.keys():
            self.limiters_dict = config['equations_limiters']
        else:
            self.limiters_dict = None
        assert set(self.enforcement_list).issubset(set(self.get_available_losses()))


    def get_available_losses(self):
        available_losses = ['X-momentum Loss', 'Y-momentum Loss' ,'Continuity Loss', 'IC Loss']
        if self.mesh.dim == 3:
            available_losses += ['Z-momentum Loss']
        return available_losses

    def balance_losses(self, method='mean'):
 
        if method == 'mean':
            if self.verbose: print([self.loss_dict[key] for key in self.enforcement_list])
            total_loss = torch.stack([self.loss_dict[key] for key in self.enforcement_list]).sum()/len(self.loss_dict.keys())
        else:
            raise NotImplementedError
        return total_loss 
    
    def get_spatial_gradients(self, solution: torch.tensor):
        '''
        We get gradients outside of the Navier-Stokes equations, as we may want to use them in the
        Soblev Norms
        '''
        gradient_dict = {}
        for channel in self.field_channel_idx_dict:
            # first derivatives
            c_grad = self.mesh.compute_derivative(solution[...,self.field_channel_idx_dict[channel]], 
                                         field_type=channel,
                                         order=1)
            
            if self.verbose: print('first derivative shape:',c_grad.shape)
            sub_dict = dict.fromkeys(gradient_str(channel,mesh_dim=self.mesh.dim, order=1))
            for i, name in zip(range(c_grad.shape[-1]),sub_dict.keys()):
                sub_dict[name] = c_grad[...,i]
            gradient_dict.update(sub_dict)
            
            if channel == 'U':
                # second derivatives
                c_grad = self.mesh.compute_derivative(c_grad, 
                                                      field_type=channel,
                                                      order=2)
                if self.verbose: print('second derivative shape:',c_grad.shape)
                sub_dict = dict.fromkeys(gradient_str(channel,mesh_dim=self.mesh.dim, order=2))
                for i, name in zip(range(c_grad.shape[-1]),sub_dict.keys()):
                    sub_dict[name] = c_grad[...,i]
                gradient_dict.update(sub_dict)
        
        return gradient_dict

    def get_temporal_gradients(self, out, time_step, input_solution=None, channel:str='U'):
        
        if time_step is None and self.time_step is not None:
            time_step = self.time_step
        elif time_step is None:
            time_step = 1
        
        dt_dict = dict.fromkeys(gradient_str(channel=channel, mesh_dim=self.mesh.dim,time_dim=True))

        if len(out.shape) == 3:
            for name in zip(dt_dict.keys()):
                dt_dict[name] = 0
        elif len(out.shape) == 4:
            dt_out = (out[:,1:,...] - out[:,:-1,...])/time_step
            dt_out = torch.nn.functional.pad(dt_out, (0,0,0,0,1,0))
            if input_solution is not None:
                print(dt_out.shape, 'compared to ', out.shape)
                dt_out[:,0,...] = (out[:,0,...] - input_solution[:,-1,...])/time_step
            else:
                # we can't calculate momentum on the first_time_step anyway
                self.mom_eqn_skip_first_ts = True
            
            for i, name in zip(self.field_channel_idx_dict[channel],dt_dict.keys()):
                dt_dict[name] = dt_out[...,i]
        else:
            raise IndexError(f'Field has {len(out.shape)} dims, when 4 (time-variant) or 3 (time-invariant/zero) is expected')

        return dt_dict
    
    def compute(self, 
                out: torch.tensor, 
                y: torch.tensor = None, 
                input_solution: torch.tensor = None, 
                time_step=None, 
                Re:torch.tensor=None) -> torch.tensor:
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
        
        # Each function below updates this dict with loss tensors
        self.loss_dict = {}
        self.device = out.device

        # Prepare Gradients
        pred_grads = self.get_spatial_gradients(out)
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
            if Re is None and self.Re is not None: 
                Re = torch.tensor(self.Re, dtype=torch.float32, device=out.device).reshape(1,1,1,1) # (batch,time,cells,channels)

            if self.verbose: print('...Attempting to calculate PDE', flush=True)
            self.compute_pde_loss(out, input_solution, time_step, pred_grads, Re)

        if self.verbose:
            print('\nLosses: Before balancing, limiting and excluding:')
            for key, value in self.loss_dict.items():
                print(' ', key, value)

        if self.limiters_dict is not None:
            self.apply_limiters()
        return self.balance_losses()

    def compute_ic_loss(self, out:torch.tensor,input_solution=None,y=None):
        if input_solution is not None:
            assert out.shape[-2] == input_solution.shape[-2]
            ic_loss = self.loss_fn(out[:,0,...], input_solution[:,-1,...])
        elif y is not None:
            ic_loss = self.loss_fn(out[:,0,...], y[:,0,...])
        else:
            raise ValueError('Either an input_solution or ground truth as inputs')
        
        self.loss_dict['IC Loss'] = ic_loss
        return ic_loss

    def compute_pde_loss(self, 
                         out:torch.tensor, 
                         input_solution:torch.tensor=None, 
                         time_step=None, 
                         grad_dict:dict=None, 
                         Re:torch.tensor=None):
        
        if grad_dict is None:
            grad_dict = self.get_spatial_gradients(out)
        
        grad_dict.update(self.get_temporal_gradients(out, time_step, input_solution))

        eqn_dict = self.pde_equations(out, 
                                      solution_index=self.field_channel_idx_dict, 
                                      Re=Re, 
                                      grad_dict=grad_dict)

        if self.mom_eqn_skip_first_ts:
            eqn_dict['X-momentum'] = eqn_dict['X-momentum'][:,1:,...]
            eqn_dict['Y-momentum'] = eqn_dict['Y-momentum'][:,1:,...]
            if self.mesh.dim == 3:
                eqn_dict['Z-momentum'] = eqn_dict['Z-momentum'][:,1:,...]

        eqn_loss_dict = dict.fromkeys(eqn_dict.keys())
        for key in eqn_loss_dict:
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
            self.loss_dict[key] = torch.max(self.loss_dict[key], torch.tensor(value,dtype=torch.float32,device=self.device))
        return

    def wandb_logging(self) -> dict:
        # if in the wandb logging we add a prefix such as 'weights/loss_func'
        # it will automatically group it in the report making it easier
        wandb_dict = {f'pde_controller/{key}': value.item() for key, value in self.loss_dict.items()}
        return wandb_dict
