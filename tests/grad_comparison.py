import torch
import numpy
import pyvista as pv

from utils.visualizer import *

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gauss_green import gaus_green_vfm_mesh

def my_cylinder_case():
    U_bc_dict = {
        'in':{
            "type":'fixedValue',
            "value":[1,0,0]
            },
        'out':{
            "type":'zeroGradient',
            },  
        'cylinder':{
            "type":'fixedValue',
            "value":[0,0,0]
            },
        'sym1':{
            "type":'symmetryPlane'
            },
        'sym2':{
            "type":'symmetryPlane'
            }   
        }
    
    p_bc_dict = {
        'in':{
            "type":'zeroGradient'
            },
        'out':{
            "type":'fixedValue',
            "value":0
            },  
        'cylinder':{
            "type":'zeroGradient'
            } ,
        'sym1':{
            "type":'symmetryPlane'
            },
        'sym2':{
            "type":'symmetryPlane'
            } 
        }
    
    return U_bc_dict, p_bc_dict


if __name__ == '__main__':
    
    plotting = True
    interactive = True
    plots = ['eqn']
    comparison = 'OpenFoam' # Can do 'VTK' but this is less accurate
    nu = 0.0133
    dt = 0.05
    '''
    NOTE: for comparison, you need to generate the files in OpenFoam, such as Grad(U) and Laplacian.
    The Laplacian in OpenFoam comes multiplied by the stress coefficient (e.g. 1/Re).
    '''

    # Get Mesh
    dir = r'C:\Users\Noahc\Downloads\c5_new_30\case.foam'
    vtk_file_reader = pv.POpenFOAMReader(dir)
    U_bc_dict, p_bc_dict = my_cylinder_case()
    bc_dict = {'U': U_bc_dict,
               'p': p_bc_dict}
    
    # init Mesh
    mesh = gaus_green_vfm_mesh(vtk_file_reader)
    mesh.add_bc_conditions(bc_dict)
    mesh.patch_face_keys_dict()
    
    # we will source the VTK mesh within the gaus_green_vfm_mesh object as it has an active pointer
    if comparison == 'OpenFoam':
        ground_truth_derivatives = mesh.mesh['grad(U)']
        ground_truth_derivatives_p = mesh.mesh['grad(p)']
        ground_truth_laplacian = (nu)*mesh.mesh['lapU']
        ground_truth_derivatives2 = np.concatenate((mesh.mesh['field0'], mesh.mesh['field1'], mesh.mesh['field2']), axis=-1) 
    elif comparison == 'VTK':
        vtk_calc = mesh.mesh.compute_derivative(scalars="U", gradient='Gradient_VTK', preference='cell')
        vtk_calc = vtk_calc.compute_derivative(scalars="Gradient_VTK", gradient='Gradient_2nd_VTK', preference='cell')
        ground_truth_derivatives = vtk_calc['Gradient_VTK']
        vtk_calc_2nd_dif = vtk_calc['Gradient_2nd_VTK']

    # get example solution
    field = torch.tensor(mesh.mesh['U'],dtype=torch.float32).unsqueeze(0)[...,:2]
    field_P = torch.tensor(mesh.mesh['p'],dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    pred_der = mesh.compute_derivative(field, field_type='U', order=1)[0,...]
    pred_der_P = mesh.compute_derivative(field_P, field_type='p', order=1)[0,...]
    pred_der_2nd = mesh.compute_derivative(pred_der.unsqueeze(0), field_type='U', order=2, original_field=field)[0,...]

    # get previous solution (for dt)
    vtk_file_reader.set_active_time_value(vtk_file_reader.time_values[-2])
    vtk_file_reader.cell_to_point_creation = False
    mesh_t0 = vtk_file_reader.read()[0]
    field_t0 = torch.tensor(mesh_t0['U'],dtype=torch.float32).unsqueeze(0)[...,:2]
    dUdt = (field - field_t0)/dt

    #['du/dxx', 'du/dyx','du/dxy', 'du/dyy', 'dv/dxx', 'dv/dyx','dv/dxy', 'dv/dyy']
    ## PREDICTION
    pred_lap = []
    pred_lap_U = (nu)*(pred_der_2nd[...,0]+pred_der_2nd[...,2])
    pred_lap_V = (nu)*(pred_der_2nd[...,4]+pred_der_2nd[...,7])
    pred_lap = torch.cat([pred_lap_U.unsqueeze(-1), pred_lap_V.unsqueeze(-1)], dim = -1)

    f0_pred = pred_der[...,0] + pred_der[...,3]
    f1_pred = dUdt[0,...,0] + field[0,...,0]*pred_der[...,0] + field[0,...,1]*pred_der[...,1] -pred_lap_U + pred_der_P[...,0]
    f2_pred = dUdt[0,...,1] + field[0,...,0]*pred_der[...,2] + field[0,...,1]*pred_der[...,3] -pred_lap_V + pred_der_P[...,1]

    print(f0_pred.shape, f1_pred.shape,f2_pred.shape, pred_lap_U.shape)
    pred_eqn = torch.cat([f0_pred.unsqueeze(-1), f1_pred.unsqueeze(-1), f2_pred.unsqueeze(-1)], dim = -1)

    ## GROUND TRUTH
    #['du/dx', 'du/dy','du/dz', 'dv/dx', 'dv/dy','dv/dz', 'dw/dx', 'dw/dy','dw/dz']
    f0_gt = ground_truth_derivatives[...,0] + ground_truth_derivatives[...,4]
    f1_gt = dUdt[0,...,0] + field[0,...,0]*ground_truth_derivatives[...,0] + field[0,...,1]*ground_truth_derivatives[...,1] -ground_truth_laplacian[...,0] + ground_truth_derivatives_p[...,0]
    f2_gt = dUdt[0,...,1] + field[0,...,0]*ground_truth_derivatives[...,3] + field[0,...,1]*ground_truth_derivatives[...,4] -ground_truth_laplacian[...,1] + ground_truth_derivatives_p[...,1]
    gt_eqn = torch.cat([torch.tensor(f0_gt).unsqueeze(-1), f1_gt.unsqueeze(-1), f2_gt.unsqueeze(-1)], dim = -1)

    # LAPLACIAN VALIDATION
    #['du/dxx', 'du/dxy', 'du/dxz','du/dyx', 'du/dyy', 'du/dyz','du/dzx', 'du/dzy', 'du/dzz',
    # 'dv/dxx', 'dv/dxy', 'dv/dxz','dv/dyx', 'dv/dyy', 'dv/dyz','dv/dzx', 'dv/dzy', 'dv/dzz',
    # 'dw/dxx', 'dw/dxy', 'dw/dxz','dw/dyx', 'dw/dyy', 'dw/dyz','dw/dzx', 'dw/dzy', 'dw/dzz'])
    ground_truth_laplacian_val_U = ground_truth_derivatives2[...,0] + ground_truth_derivatives2[...,4] +  ground_truth_derivatives2[...,8] 
    ground_truth_laplacian_val_V = ground_truth_derivatives2[...,9] + ground_truth_derivatives2[...,13] +  ground_truth_derivatives2[...,17]
    gt_lap_val = nu*torch.cat([torch.tensor(ground_truth_laplacian_val_U).unsqueeze(-1), torch.tensor(ground_truth_laplacian_val_V).unsqueeze(-1)], dim = -1)

    # Get the directory where the current script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir_base = '\\'.join(os.path.split(script_dir))+'/results'
    
    # First derivatives:
    if plotting:
        if '1st' in plots:
            results_dir = results_dir_base + '/first_derivatives_v3'
            if mesh.dim == 3:
                names = ['du/dx', 'du/dy', 'du/dz','dv/dx', 'dv/dy', 'dv/dz', 'dw/dx', 'dw/dy', 'dw/dz']
            else:
                names = ['du/dx', 'du/dy','dv/dx', 'dv/dy']
            for i, j in zip(range(pred_der.shape[-1]), [0,1,3,4]):
                plot_comparison(mesh.mesh, results_dir, f'Cylinder_{names[i]}',
                                ground_truth = ground_truth_derivatives,
                                prediction = pred_der.detach().numpy(),
                                i=i,j=j, interactive = interactive)
    
    # Second derivatives:
        if 'Lap' in plots:
            results_dir = results_dir_base + '/second_derivatives_v3'
            names = ['Lap_U', 'Lap_V']
            for i in range(pred_lap.shape[-1]):
                plot_comparison(mesh.mesh, results_dir, f'Cylinder_{names[i]}',
                                ground_truth = ground_truth_laplacian,
                                prediction = pred_lap.detach().numpy(),
                                i=i, interactive = interactive)
        
        if '2nd' in plots:
            results_dir = results_dir_base + '/second_derivatives_v3/all'
            indices_3d = np.array([0,4,9,13])
            
            if mesh.dim == 3:
                indices_2d = indices_3d
            else:
                #['du/dxx', 'du/dyx','du/dxy', 'du/dyy', 'dv/dxx', 'dv/dyx','dv/dxy', 'dv/dyy']
                indices_2d = np.array([0,3,4,7])
            names = np.array(['du/dxx', 'du/dxy', 'du/dxz','du/dyx', 'du/dyy', 'du/dyz','du/dzx', 'du/dzy', 'du/dzz',
                                'dv/dxx', 'dv/dxy', 'dv/dxz','dv/dyx', 'dv/dyy', 'dv/dyz','dv/dzx', 'dv/dzy', 'dv/dzz',
                                'dw/dxx', 'dw/dxy', 'dw/dxz','dw/dyx', 'dw/dyy', 'dw/dyz','dw/dzx', 'dw/dzy', 'dw/dzz'])
            print(ground_truth_derivatives2.shape, pred_der_2nd.shape)
            for i,j in zip(indices_2d, indices_3d):
                plot_comparison(mesh.mesh, results_dir, f'Cylinder_{names[j]}',
                                ground_truth = ground_truth_derivatives2,
                                prediction = pred_der_2nd.detach().numpy(),
                                i=i, j=j, interactive = interactive)
        
        if 'eqn' in plots:
            results_dir = results_dir_base + '/equation_losses/all'
            indices_3d = np.array([0,4,9,13])
            
            indices_2d = np.array([0,3,4,7])
            names = np.array(['Continuity', 'Momentum-X', 'Momentum-Y'])
            print(ground_truth_derivatives2.shape, pred_der_2nd.shape)
            for i in range(len(names)):
                plot_comparison(mesh.mesh, results_dir, f'Cylinder_{names[i]}',
                                ground_truth = gt_eqn.numpy(),
                                prediction = pred_eqn.detach().numpy(),
                                i=i, interactive = interactive)
                
        if 'lap validation' in plots:
            results_dir = results_dir_base + '/second_derivatives_v3/lap_val'
            indices_3d = np.array([0,4,9,13])
            
            indices_2d = np.array([0,3,4,7])
            names = np.array(['Lap_U', 'Lap_V'])
            print(ground_truth_derivatives2.shape, pred_der_2nd.shape)
            for i in range(len(names)):
                plot_comparison(mesh.mesh, results_dir, f'Cylinder_{names[i]}',
                                ground_truth = ground_truth_laplacian,
                                prediction = gt_lap_val.detach().numpy(),
                                i=i, interactive = interactive)