import torch
import numpy
import pyvista as pv
import matplotlib.pyplot as plt
import Ofpp

from utils.visualizer import *

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gauss_green import gaus_green_vfm_mesh
from grad_comparison import my_cylinder_case
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
    #print(mesh.mesh.GetCell(2300))

    #raise KeyboardInterrupt
    openfoam_volume = Ofpp.parse_internal_field(r'C:\Users\Noahc\Downloads\c5_new_30\30\V')
    plot_comparison(mesh.mesh, None, f'Volume Difference',
                        prediction=mesh.cell_volume[...,None],
                        ground_truth = openfoam_volume[...,None],
                        i=0, interactive = interactive)
    
    # import csv
    # with open('output.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(mesh.unique_faces)   
    
    openfoam_coords = Ofpp.parse_internal_field(r'C:\Users\Noahc\Downloads\c5_new_30\30\C')
    
    #x_dif = openfoam_coordsx - mesh.cell_coords[:,0]
    #y_dif = openfoam_coordsy - mesh.cell_coords[:,1]

    
    # plot_comparison(mesh.mesh, None, f'x_coords Difference',
    #                     prediction=openfoam_coords,
    #                     ground_truth = mesh.cell_coords,
    #                     i=0, interactive = interactive)
    
    # plot_comparison(mesh.mesh, None, f'y_coords Difference',
    #                     prediction=openfoam_coords,
    #                     ground_truth = mesh.cell_coords,
    #                     i=1, interactive = interactive)

    #raise KeyboardInterrupt
    
    mesh.add_bc_conditions(bc_dict)
    mesh.patch_face_keys_dict()

    field = torch.tensor(mesh.mesh['U'],dtype=torch.float32).unsqueeze(0)[...,:2]
    field.shape
    #phi = mesh.compute_phi(field) # 18180 faces 18225 square (pad by 45)
    # phi = torch.nn.functional.pad(input=phi, pad=(0, 45), mode='constant', value=0)
    # phi = phi.reshape(135,135)

    div_field, lap_field, phi = mesh.compute_everything(field)
    div_field, lap_field = div_field.squeeze(0).squeeze(0), lap_field.squeeze(0).squeeze(0)*nu

    gt_div = mesh.mesh['divU']
    gt_lap = mesh.mesh['lapU']
    

    names = ['ududx + vdudy', 'udvdx + vdvdy']
    for i in range(len(names)):
        #break
        plot_comparison(mesh.mesh, None, f'Cylinder_{names[i]}',
                        ground_truth = gt_div,
                        prediction = div_field.numpy(),
                        i=i, interactive = interactive)
        
    
    
    names = ['Lapx', 'Lapy']
    for i in range(len(names)):
        break
        plot_comparison(mesh.mesh, None, f'Cylinder_{names[i]}',
                        ground_truth = gt_lap,
                        prediction = lap_field.numpy(),
                        i=i, interactive = interactive,
                        clims=[-1,1])




    phi_gt = Ofpp.parse_field_all(r'C:\Users\Noahc\Downloads\c5_new_30\30\phi')[0]
    phi_dif = torch.tensor(phi_gt) - phi[0,0,mesh.internal_faces_idx]
    phi_red_flag_indices = torch.nonzero(phi_dif > 1e-01, as_tuple=True)

    cell_idx = mesh.owner[mesh.internal_faces_idx[phi_red_flag_indices]]
    phi_mesh_plot = torch.zeros_like(lap_field)[...,0]
    phi_mesh_plot[cell_idx] = 1
    phi_mesh_plot = phi_mesh_plot.unsqueeze(-1)

    plot_comparison(mesh.mesh, None, f'Phi Difference',
                        prediction=phi_mesh_plot,
                        ground_truth = phi_mesh_plot,
                        i=0, interactive = interactive)