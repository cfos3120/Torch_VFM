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
    comparison = 'OpenFoam' # Can do 'VTK' but this is less accurate
    nu = 0.0133
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
        ground_truth_laplacian = mesh.mesh['lapU']
        ground_truth_derivatives2 = np.concatenate((mesh.mesh['field0'], mesh.mesh['field1'], mesh.mesh['field2']), axis=-1) 
    elif comparison == 'VTK':
        vtk_calc = mesh.mesh.compute_derivative(scalars="U", gradient='Gradient_VTK', preference='cell')
        vtk_calc = vtk_calc.compute_derivative(scalars="Gradient_VTK", gradient='Gradient_2nd_VTK', preference='cell')
        ground_truth_derivatives = vtk_calc['Gradient_VTK']
        vtk_calc_2nd_dif = vtk_calc['Gradient_2nd_VTK']

    # get example solution
    field = torch.tensor(mesh.mesh['U'],dtype=torch.float32).unsqueeze(0)[...,:2]
    pred_der = mesh.compute_derivative(field, field_type='U', order=1)[0,...]
    pred_der_2nd = mesh.compute_derivative(pred_der.unsqueeze(0), field_type='U', order=2, original_field=field)[0,...]

    # Calculate Laplacian: 3D
    #['du/dxx', 'du/dyx', 'du/dzx','du/dxy', 'du/dyy', 'du/dzy','du/dxz', 'du/dyz', 'du/dzz',
    # 'dv/dxx', 'dv/dyx', 'dv/dzx','dv/dxy', 'dv/dyy', 'dv/dzy','dv/dxz', 'dv/dyz', 'dv/dzz',
    # 'dw/dxx', 'dw/dyx', 'dw/dzx','dw/dxy', 'dw/dyy', 'dw/dzy','dw/dxz', 'dw/dyz', 'dw/dzz']
    # 2D:
    #['du/dxx', 'du/dyx','du/dxy', 'du/dyy',
    # 'dv/dxx', 'dv/dyx','dv/dxy', 'dv/dyy']
    pred_lap = []
    exclude_3d = False

    if mesh.dim == 3:
        for i in [0,9,18]:
            if exclude_3d:
                lap = pred_der_2nd[:,i] + pred_der_2nd[:,i+4]
            else: 
                lap = pred_der_2nd[:,i] + pred_der_2nd[:,i+4] + pred_der_2nd[:,i+8]
            pred_lap.append(lap.unsqueeze(-1))
    else:
        for i in [0,4]:
            lap = pred_der_2nd[:,i] + pred_der_2nd[:,i+3]
            pred_lap.append(lap.unsqueeze(-1))
    pred_lap = torch.cat(pred_lap, dim=-1)#*nu
    
    # Get the directory where the current script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir_base = '\\'.join(os.path.split(script_dir))+'/results'
    
    # First derivatives:
    if plotting:
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
        results_dir = results_dir_base + '/second_derivatives_v3'
        names = ['Lap_U', 'Lap_V', 'Lap_W']
        for i in range(pred_lap.shape[-1]):
            break
            plot_comparison(mesh.mesh, results_dir, f'Cylinder_{names[i]}',
                            ground_truth = pred_lap.detach().numpy(),
                            prediction = pred_lap.detach().numpy(),
                            i=i, interactive = interactive)
        
        results_dir = results_dir_base + '/second_derivatives_v3/all'
        indices_3d = np.array([0,4,9,13])
        #['du/dxx', 'du/dyx','du/dxy', 'du/dyy',
        # 'dv/dxx', 'dv/dyx','dv/dxy', 'dv/dyy']
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