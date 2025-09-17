import torch
from .gauss_green import *
from .physics.operators import *
from .utils.data_utils import get_vtk_file_reader

class sobolev_controller():
    def __init__(self, config: dict, device, space_dim:int=2) -> None:
        self.config = config
        file_name = config['dataset_params']['mesh_pointer']
        
        self.dim = space_dim
        
        self.loss_function = torch.nn.MSELoss()
        if config['parameters']['loss_fn_pred'] == 'h1':
            self.sobolev_order = 1
        elif config['parameters']['loss_fn_pred'] == 'h2':
            self.sobolev_order = 2
            raise NotImplementedError('Sobolev order 2 not implemented yet')
        else:
            self.sobolev_order = 0

        # Weighting
        if 'h_weighting' in config['parameters'].keys():
            self.h_w = config['parameters']['h_weighting']
        else:
            self.h_w = [1,1,1]

        self.vtk_file_reader = get_vtk_file_reader(file_name)
        self.mesh = gaus_green_vfm_mesh(self.vtk_file_reader, 
                                        L=1, 
                                        device=device, 
                                        bc_dict=get_bc_dict(type=config['dataset_params']['bc_dict_type'])
                                        )
        
        self.w = self.mesh.mesh.cell_volumes.reshape(1,1,-1,1)
        

    def __call__(self, out: torch.tensor, y: torch.tensor):
        
        loss_dict = {}
        if self.dim == 2:
            idx_u = [0,1,3,4]
            idx_p = [0,1]
            channel_dim = 3
        else:
            idx_u = [0,1,2,3,4,5,6,7,8]
            idx_p = [0,1,2]
            channel_dim = 4

        h = self.h_w[0]*(out - y)**2
        h_den = self.h_w[0]*y**2

        loss_dict['training/H0_norm_loss'] =   torch.mean(torch.sqrt(torch.sum(self.w*h,dim=-2))/torch.sqrt(torch.sum(self.w*h_den,dim=-2))).item()

        if self.sobolev_order >= 1:

            U, p = out[..., :self.dim], out[..., [-1]]
            U_y, p_y = y[..., :self.dim], y[..., [-1]]
             
            if self.dim == 2:
                U = torch.nn.functional.pad(U,(0,1))
                U_y = torch.nn.functional.pad(U_y,(0,1))

            _, u_grad = Divergence_Operator.caclulate(self.mesh,U,field_type = 'U')
            _, p_grad = Divergence_Operator.caclulate(self.mesh,p,field_type = 'p')
            _, u_grad_den = Divergence_Operator.caclulate(self.mesh,U_y,field_type = 'U')
            _, p_grad_den = Divergence_Operator.caclulate(self.mesh,p_y,field_type = 'p')

        
            sol_grad = torch.cat((u_grad[...,idx_u],p_grad[...,idx_p]),dim=-1)
            sol_grad_den = torch.cat((u_grad_den[...,idx_u],p_grad_den[...,idx_p]),dim=-1)
            
            h += self.h_w[1]*(sol_grad - sol_grad_den).view(*u_grad.shape[:-1], channel_dim, self.dim).sum(dim=-1)**2
            h_den += self.h_w[1]*(sol_grad_den).view(*u_grad.shape[:-1], channel_dim, self.dim).sum(dim=-1)**2

            loss_dict['training/H1_norm_loss'] =   torch.mean(torch.sqrt(torch.sum(self.w*h,dim=-2))/torch.sqrt(torch.sum(self.w*h_den,dim=-2))).item()

        if self.sobolev_order == 2:
            raise NotImplementedError('Sobolev order 2 not implemented yet')
        
        h = torch.sum(self.w*h, dim=-2)
        h_den = torch.sum(self.w*h_den, dim=-2)

        hn_loss = torch.mean(torch.sqrt(h)/torch.sqrt(h_den))
        loss_dict['training/Hn_enforced_loss'] = hn_loss.item()

        return hn_loss, loss_dict