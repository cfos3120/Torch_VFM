import pyvista as pv
import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(mesh, file_location, file_name, ground_truth, prediction, i, j=None, interactive = True, clims = None, clims2 = None, show_mesh =False):
    if j is None:
        j = i 

    plotter = pv.Plotter(shape=(1, 3),off_screen= not interactive)
    #get copy of mesh
    mesh_for_plotting = mesh.mesh.copy(deep=True)
    mesh_for_plotting['Torch FVM'] = prediction[:,i]
    mesh_for_plotting['Ground Truth'] = ground_truth[:,j]
    mesh_for_plotting['Difference'] = ground_truth[:,j] - prediction[:,i]

    # comparison colour map:
    berlin_cmap = plt.get_cmap("berlin")
    if clims2 is None:
        abs_max = np.max(np.abs(mesh_for_plotting['Difference']))
        clims2 = [-abs_max, abs_max]

    if clims is None:
        abs_max = np.max(np.abs(mesh_for_plotting['Ground Truth']))
        clims = [-abs_max, abs_max]

    # Plot scalar1 in the first subplot
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh_for_plotting, 
                     scalars='Torch FVM', 
                     cmap='viridis', 
                     clim=clims, 
                     show_scalar_bar=True, 
                     show_edges=show_mesh, 
                     edge_opacity=0.25,
                     copy_mesh=True)
    plotter.add_title("Torch FVM")
    plotter.view_xy()

    # Plot scalar2 in the second subplot
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh_for_plotting, 
                     scalars='Ground Truth', 
                     cmap='viridis', 
                     clim=clims, 
                     show_scalar_bar=True, 
                     show_edges=show_mesh, 
                     edge_opacity=0.25,
                     copy_mesh=True)
    plotter.add_title("Ground Truth")
    plotter.view_xy()

    # Plot scalar3 in the third subplot
    plotter.subplot(0, 2)
    plotter.add_mesh(mesh_for_plotting, 
                     scalars='Difference', 
                     cmap=berlin_cmap, 
                     clim=clims2, 
                     show_scalar_bar=True, 
                     show_edges=show_mesh, 
                     edge_opacity=0.25,
                     copy_mesh=True)
    plotter.add_title(f"Difference")
    plotter.view_xy()

    # Display the plot
    if interactive:
        plotter.show()
    else:
        plotter.show(auto_close=False)
        plt.imsave(f"{file_location}/{file_name.replace('/','_')}.png", plotter.image)
        plotter.close()