import torch
import numpy as np
import pyvista as pv
import Ofpp
from utils.visualizer import *
pv.set_jupyter_backend('static') 

import sys
import os
sys.path.insert(0, r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\Torch_VFM')
from src.utils.mesh_utils import *
from src.utils.data_utils import get_bc_dict
from src.gauss_green import gaus_green_vfm_mesh
from src.physics.operators import *

def get_bc_dict():
    U_bc_dict = {
                'inlet':{ "type":'fixedValue', "value":[1,0,0] },
                'outlet':{ "type":'zeroGradient'},  
                'cylinder':{ "type":'noSlip'},  
                'frontAndBack':{ "type":'empty'}, 
                }
    p_bc_dict = {
        'inlet':{ "type":'zeroGradient' },
        'outlet':{ "type":'zeroGradient'},  
        'cylinder':{ "type":'zeroGradient'},  
        'frontAndBack':{ "type":'empty' } ,
        }
    return {'U':U_bc_dict, 'p':p_bc_dict}

dir = r'C:\Users\Noahc\Documents\USYD\tutorial\cylinder_steady\case.foam'
vtk_file_reader = pv.POpenFOAMReader(dir)
dtype = torch.float32

# Get Mesh
mesh = gaus_green_vfm_mesh(vtk_file_reader, dtype=dtype, bc_dict=get_bc_dict())
#mesh.add_bc_conditions(get_bc_dict())

nu = 0.025
Re = 1/nu 
sample_solution_U = mesh.vtk_mesh.cell_data['U']
sample_solution_p = mesh.vtk_mesh.cell_data['p']

sample_u = torch.tensor(sample_solution_U[...,:2],dtype=dtype).reshape(-1,2).unsqueeze(0).unsqueeze(0)
sample_u = torch.nn.functional.pad(sample_u, (0, 1))
sample_p = torch.tensor(sample_solution_p,dtype=dtype).reshape(-1,1).unsqueeze(0).unsqueeze(0)

_, gradp_pred = Divergence_Operator.caclulate(mesh, field=sample_p, field_type='p')
divU_pred, gradU_pred = Divergence_Operator.caclulate(mesh, field=sample_u)
lap_pred = Laplacian_Operator.caclulate(mesh, field=sample_u, correction_method=mesh.correction_method, gradient_field=gradU_pred)

mom_pred = divU_pred - (1/Re)*lap_pred + gradp_pred

gt_div = torch.tensor(mesh.vtk_mesh.cell_data['divU_phiAll'])
gt_lap = torch.tensor(mesh.vtk_mesh.cell_data['lapU'])
gt_gradu = torch.tensor(mesh.vtk_mesh.cell_data['grad(U)'])
gt_gradp = torch.tensor(mesh.vtk_mesh.cell_data['grad(p)'])

plot_comparison(mesh.vtk_mesh, None, f'None',i=0, interactive = True, point_label=False,
                    prediction = lap_pred.squeeze(0).squeeze(0),
                    ground_truth = gt_lap.squeeze(0).squeeze(0),
                    clims=[-10,10],
                    clims2=[-10,10],
                    zoom=True
                    )