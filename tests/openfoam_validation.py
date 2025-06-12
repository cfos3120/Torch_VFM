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
    vtk_file_reader.set_active_time_value(vtk_file_reader.time_values[-1])
    vtk_file_reader.cell_to_point_creation = False
    mesh = vtk_file_reader.read()[0]
    field = mesh['U'][...,:2]
    
    gt_derivatives = mesh['grad(U)']
    gt_derivatives_p = mesh['grad(p)']
    gt_laplacian = (nu)*mesh['lapU']
    gt_derivatives2 = np.concatenate((mesh['field0'], mesh['field1'], mesh['field2']), axis=-1)
    gt_dudt = mesh['ddtU']
    gt_div = mesh['divU']
    gt_eqn = mesh['UEqn']

    # get previous solution (for dt)
    vtk_file_reader.set_active_time_value(vtk_file_reader.time_values[-2])
    vtk_file_reader.cell_to_point_creation = False
    mesh_t0 = vtk_file_reader.read()[0]
    

    ''' 
      0   |   1   |   2   |   3   |   4   |   5   |   6   |   7   |   8
    ---------------------------------------------------------------------      
    du/dx | du/dy | du/dz | dv/dx | dv/dy | dv/dz | dw/dx | dw/dy | dw/dz
    '''

    field_t0 = mesh_t0['U'][...,:2]
    
    # Euler
    dUdt = (field - field_t0)/dt
    
    # Backward
    dUdt = (3*field - 4*field_t0 + field_t0)/(2*dt) #<- Matches

    div_field = np.zeros_like(field)
    div_field[...,0] = field[...,0]*gt_derivatives[...,0] + field[...,1]*gt_derivatives[...,1]
    div_field[...,1] = field[...,0]*gt_derivatives[...,3] + field[...,1]*gt_derivatives[...,4]
    # the reason why this does not work is because the fvc::div(phi,U) function in OpenFoam, weights the divergence
    # based on the face fluxes

    # Navier Stokes
    eqn = dUdt[...,:2] + div_field - nu*gt_derivatives2[...,:2] + gt_derivatives_p[...,:2]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = '\\'.join(os.path.split(script_dir))+'/results/OF_validation' 
    
    # ddt
    names = ['dudt', 'dvdt']
    for i in range(len(names)):
        break
        plot_comparison(mesh, results_dir, f'Cylinder_{names[i]}',
                        ground_truth = gt_dudt[...,:2],
                        prediction = dUdt,
                        i=i, interactive = interactive)
    
    
    names = ['ududx + vdudy', 'udvdx + vdvdy']
    for i in range(len(names)):
        #break
        plot_comparison(mesh, results_dir, f'Cylinder_{names[i]}',
                        ground_truth = gt_div,
                        prediction = div_field,
                        i=i, interactive = interactive)
        
    names = ['X-momentum', 'Y-momentum']
    for i in range(len(names)):
        break
        plot_comparison(mesh, results_dir, f'Cylinder_{names[i]}',
                        ground_truth = gt_eqn,
                        prediction = eqn,
                        i=i, interactive = interactive)