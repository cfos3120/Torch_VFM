import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gauss_green import gaus_green_vfm_mesh
from src.pde_controller import pde_controller

if __name__ == '__main__':

    # Control Arguments:
    model_ckpt_path = None
    epochs = 1

    # Save Path:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = '\\'.join(os.path.split(script_dir))
    save_dir = base_dir+'/results/mock_model'

    # Initialize PDE Controller
    config = {'mesh_file_pointer': base_dir+'\open_foam_case\case.foam',
              'enforcement_list': ['Continuity Loss', 'IC Loss'],
              'equations_limiters' : {'Continuity Loss' : 0.001
                  },
              'dataset_channels' : {'U': [0,1],
                                    'p': [2]
                                    },
              'pde_equation' : 'navier_stokes_2d',
              'settings': {
                'verbose':                False,
                'ic_loss':                True,
                'pin_first_ts':           True,
                'pde_loss':               True,
                'mom_eqn_skip_first_ts':  False,
                'soblov_norms':           False   # Not Implemented yet
                },
               'enforcement_list': ['IC Loss'],
               'equations_limiters': {}
            }
    
    print('Creating PDE Controller')
    pde_controller = pde_controller(config=config)
    print('PDE Controller created')

    # In this case the mesh has already an openfoam solution, so we can use that instead of creating
    # a dataloader -> Remember Data is already in 3d
    file_reader = pde_controller.vtk_file_reader

    # at t0 - Input Solution (batch, time, cells, channels)
    file_reader.set_active_time_value(file_reader.time_values[-1])
    mesh = file_reader.read()[0]
    U = torch.tensor(mesh['U'],dtype=torch.float32)[...,:2] # for 2D
    p = torch.tensor(mesh['p'],dtype=torch.float32).unsqueeze(-1)
    x_i = torch.cat([U,p],dim=-1).unsqueeze(0)

    # at t1 - Output Solution (batch, time, cells, channels)
    file_reader.set_active_time_value(file_reader.time_values[-1])
    mesh = file_reader.read()[0]
    U = torch.tensor(mesh['U'],dtype=torch.float32)[...,:2] # for 2D
    p = torch.tensor(mesh['p'],dtype=torch.float32).unsqueeze(-1)
    y = torch.cat([U,p],dim=-1).unsqueeze(0)

    # Model Inputs (batch, cells, space dims)
    x = torch.tensor(mesh.cell_centers().points,dtype=torch.float32).unsqueeze(0)[...,:2]
    print('Dataset created')
    
    # Initialize a Network
    model = torch.nn.Sequential(torch.nn.Linear(2, 64),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(64, 128),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(128, 128),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(128, 128),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(128, 128),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(128, 128),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(128, 128),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(128, 64),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(64, 3),
                                     torch.nn.ReLU()
                                     )
    print('Model initialized')

    if model_ckpt_path:
        # Load Checkpoint and Evaluate Immediately
        checkpoint = torch.load(model_ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'loaded model weights from {model_ckpt_path}')
    else:
        # Train new model
        print(f'Training fresh new model for {epochs} epochs')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_function = torch.nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=1000)

        for epoch in range(epochs):
            optimizer.zero_grad()  # Clear gradients
            out = model(x).unsqueeze(0)
            loss = loss_function(out, y)
            eqn_loss = pde_controller.compute(out, y, input_solution=x_i)
            loss += eqn_loss
            loss.backward()       # Compute gradients
            optimizer.step()      # Update weights using Adam
            scheduler.step()
            print(f'epoch: {epoch:3} loss: {loss.item():6.4}')  

        checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': loss
            }
        
        os.makedirs(save_dir, exist_ok=True)
        torch.save(checkpoint, save_dir+'model_checkpoint.pth')