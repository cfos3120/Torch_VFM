import pyvista as pv
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def scalar_bar(plotter, actor):
    plotter.add_scalar_bar(
        title = actor,
        n_labels=2,  # Only show min and max
        width=0.75,   # Slim width
        height=0.1,  # Tall height
        position_x=0.1,  # Right side
        position_y=0.1,  # Centered vertically
        vertical=False,
        fmt="%.5g",  # One decimal place
        label_font_size=11,
        title_font_size=12,
        shadow=True
    )

def plot_comparison(mesh, file_location, file_name, ground_truth, prediction, i, j=None, interactive = True, clims = None, clims2 = None, show_mesh =False):
    if j is None:
        j = i 

    plotter = pv.Plotter(shape=(1, 3),
                         off_screen= not interactive, 
                         window_size=[1000, 400])
    #get copy of mesh
    mesh_for_plotting = mesh.copy(deep=True)
    mesh_for_plotting['Torch FVM'] = prediction[:,i]
    mesh_for_plotting['Ground Truth'] = ground_truth[:,j]
    mesh_for_plotting['Difference'] = ground_truth[:,j] - prediction[:,i]

    # comparison colour map:
    berlin_cmap = plt.get_cmap("bwr")
    if clims2 is None:
        abs_max = np.max(np.abs(mesh_for_plotting['Difference']))
        clims2 = [-abs_max, abs_max]

    if clims is None:
        abs_max = np.max(np.abs(mesh_for_plotting['Ground Truth']))
        clims = [-abs_max, abs_max]

    # Plot scalar1 in the first subplot
    plotter.subplot(0, 0)
    actor = plotter.add_mesh(mesh_for_plotting, 
                     scalars='Torch FVM', 
                     cmap='viridis', 
                     clim=clims, 
                     show_scalar_bar=False, 
                     show_edges=show_mesh, 
                     edge_opacity=0.25,
                     copy_mesh=True)
    plotter.add_title("Torch FVM")
    plotter.view_xy()
    # Custom scalar bar
    scalar_bar(plotter, 'Torch FVM')

    # Plot scalar2 in the second subplot
    plotter.subplot(0, 1)
    actor = plotter.add_mesh(mesh_for_plotting, 
                     scalars='Ground Truth', 
                     cmap='viridis', 
                     clim=clims, 
                     show_scalar_bar=False, 
                     show_edges=show_mesh, 
                     edge_opacity=0.25,
                     copy_mesh=True)
    plotter.add_title("Ground Truth")
    plotter.view_xy()
    # Custom scalar bar
    scalar_bar(plotter, 'Ground Truth')


    # Plot scalar3 in the third subplot
    plotter.subplot(0, 2)
    actor = plotter.add_mesh(mesh_for_plotting, 
                     scalars='Difference', 
                     cmap=berlin_cmap, 
                     clim=clims2, 
                     show_scalar_bar=False, 
                     show_edges=show_mesh, 
                     edge_opacity=0.25,
                     copy_mesh=True)
    plotter.add_title(f"Difference")
    plotter.view_xy()
    # Custom scalar bar
    scalar_bar(plotter, "Difference")
    


    # Display the plot
    if interactive:
        plotter.show()
    else:
        plotter.show(auto_close=False)
        
        os.makedirs(file_location, exist_ok=True)
        print('saving in directory: ', file_location)
        plt.imsave(f"{file_location}/{file_name.replace('/','_')}.png", plotter.image)
        plotter.close()