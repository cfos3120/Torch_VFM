import torch
import vtk
import numpy
from .utils.mesh_utils import *
from .boundary_conditions import get_boundary_flux

class gaus_green_vfm_mesh():
    def __init__(self, vtk_file_reader) -> None:
        
        # Read in Mesh in VTK
        vtk_file_reader.set_active_time_value(vtk_file_reader.time_values[-1])
        vtk_file_reader.cell_to_point_creation = False
        vtk_file_reader.enable_all_patch_arrays()
        self.mesh = vtk_file_reader.read()[0]

        # Mesh readily available components
        self.points = np.array(self.mesh.points)
        self.n_cells = self.mesh.n_cells
        self.cell_coords = self.mesh.cell_centers().points
        self.dim = detect_dimension(self.cell_coords)
        self.mesh = self.mesh.compute_cell_sizes()
        self.cell_volume = self.mesh["Volume"]

        self.device = torch.device('cpu')

        # Custom face owner/neighbour indices and area/normals
        self.fetch_faces()
        self.cell2cell_face_intercepts()
        self.convert_to_tensor(dtype=torch.float32)

        try:
            # TODO: replace this with VTK Boundary finder
            self.boundaries = vtk_file_reader.read()['boundary']
        except:
            self.boundaries = None
            print('No Boundary Patches Found')
            
        # Area Vectors for FVM Calculation
        self.compute_face_area_vectors()
    
    def convert_to_tensor(self, dtype=torch.float32) -> None:
        self.owner          = torch.tensor(self.owner)        
        self.neighbour       = torch.tensor(self.neighbour)
        self.face_areas     = torch.tensor(self.face_areas, dtype = dtype)
        self.face_normals   = torch.tensor(self.face_normals, dtype = dtype)
        self.cell_volumes   = torch.tensor(self.cell_volume, dtype = dtype)
        self.interp_ws      = torch.tensor(self.interp_ws, dtype = dtype).reshape(1,1,-1,1)

    def to(self, device: torch.device) -> None:
        self.device = device
        self.owner.to(device)    
        self.neighbour.to(device)
        self.face_areas.to(device)
        self.face_normals.to(device)
        self.cell_volumes.to(device)
        self.interp_ws.to(device)
        
    def fetch_faces(self) -> None:
        self.faces_dict = {}
        self.owner = []
        self.neighbour = []
        self.face_areas = []
        self.face_normals = []
        self.face_centres = []
        
        # For each cell, get faces, then for each face get points and calculate area/normals
        for cell_id in range(self.n_cells):
            vtk_cell = self.mesh.GetCell(cell_id)
            n_faces = vtk_cell.GetNumberOfFaces()

            for face_id in range(n_faces):
                face = vtk_cell.GetFace(face_id)
                face_points = [face.GetPointId(i) for i in range(face.GetNumberOfPoints())]
                face_key = tuple(sorted(face_points))

                if face_key not in self.faces_dict:
                    self.faces_dict[face_key] = len(self.owner)
                    self.owner.append(cell_id)
                    self.face_centres.append(self.points[face_points,:].mean(axis=0))
                    self.neighbour.append(-1)
                    self.face_areas.append(compute_face_area(face_points, self.points))
                    self.face_normals.append(compute_face_normal(face_points, self.points))
                else:
                    idx = self.faces_dict[face_key]
                    self.neighbour[idx] = cell_id

        self.unique_faces = list(self.faces_dict.keys())

        # get boundary index
        self.boundary_faces_idx = np.where(np.array(self.neighbour) == -1)[0]
        self.internal_faces_idx = np.where(np.array(self.neighbour) != -1)[0]

        # convert to numpy
        self.owner          = np.array(self.owner)
        self.neighbour      = np.array(self.neighbour)
        self.face_normals   = np.array(self.face_normals)[...,:self.dim]
        self.face_areas     = np.array(self.face_areas)
        self.cell_volume    = np.array(self.cell_volume)
    
    def cell2cell_face_intercepts(self) -> None:
        
        self.face_intersects    = np.zeros_like(self.face_normals)
        self.interp_ws      = np.zeros_like(self.face_areas)

        # Internal Faces cell2cell intersections
        for face_key in self.internal_faces_idx:
            p1, p2, p3 = [self.points[p_idx] for p_idx in self.unique_faces[face_key][:3]]
            A, B, C, D = plane_equation_from_points(p1, p2, p3)

            # Line direction vector
            
            owner_cell_coord = self.cell_coords[self.owner[face_key]] 
            neighbour_cell_coord = self.cell_coords[self.neighbour[face_key]]
            line_dir = owner_cell_coord - neighbour_cell_coord

            denominator = A * line_dir[0] + B * line_dir[1] + C * line_dir[2]
            t = -(A * owner_cell_coord[0] + B * owner_cell_coord[1] + C * owner_cell_coord[2] + D) / denominator
            
            intersect_coord = owner_cell_coord + t * line_dir
            interp_w = np.linalg.norm(intersect_coord - neighbour_cell_coord) / np.linalg.norm(owner_cell_coord - neighbour_cell_coord)
            
            self.face_intersects[face_key,:] = intersect_coord[:self.dim]
            self.interp_ws[face_key] = interp_w
            
        print(f'Calculating Cell2Cell at Face Linear Interpolation Weights (L2):\n  min w:{np.min(self.interp_ws[self.internal_faces_idx]):.4f}, \
              max w:{np.max(self.interp_ws[self.internal_faces_idx]):.4f}, \
              mean w:{np.mean(self.interp_ws[self.internal_faces_idx]):.4f}')
        
        # TODO: calculate difference from intercept to face centre as %, display min, max, mean
        print(f'Assessing Linear Intercept Skew From Face Centroid:\n TBD...')

    def patch_face_keys_dict(self, exclusion_list:list=[]) -> None:
        print('Collocating Boundary Patches to Cell Faces')

        self.patch_face_keys = dict.fromkeys(self.boundaries.keys())

        for patch in self.patch_face_keys:
            patch_face_idx = self.find_bc_face_idx(patch)
            
            if patch_face_idx is None or patch in exclusion_list:
                print(f'Excluding Z-axis patch {patch}')
                exclusion_list.append(patch)
            else:
                self.patch_face_keys[patch] = np.array(patch_face_idx)

        for patch in set(exclusion_list):
            del self.patch_face_keys[patch]
        
    
    def find_bc_face_idx(self, patch) -> tuple:
        assert len(self.boundary_faces_idx) > 0

        patch_points = np.array(self.boundaries[patch].points)
        if self.dim == 2 and np.all(patch_points[:, -1] == patch_points[0, -1], axis=0):
            return None
            
        patch_point_indices = find_indices_dict(patch_points, self.points)
        
        patch_face_keys= []
        for face_key in self.boundary_faces_idx:
            if np.all(np.isin([x for x in self.unique_faces[face_key]], patch_point_indices)):
                patch_face_keys.append(face_key)
        
        if patch_face_keys == []:
            raise KeyError(f'Patch {patch} was not found in the face list')
        else:
            print(f' Found Patch "{patch}" with {len(patch_face_keys)} Faces')

        return patch_face_keys
    
    def compute_face_area_vectors(self) -> None:
        self.Sf = (self.face_areas.reshape(1,-1,1) * self.face_normals) # (n_faces, 3)

    def compute_unweighted_gradient(self, field_values:torch.tensor) -> torch.tensor:

        

        assert len(field_values.shape) == 4
        assert field_values.shape[2] == self.n_cells
        assert self.device == field_values.device

        batch_size = field_values.shape[0]
        channels = field_values.shape[-1]
        space_dim = self.Sf.shape[-1]
        time_dim = field_values.shape[1]
        
        # Init Gradient Field:
        grad_field = torch.zeros((batch_size, time_dim, self.n_cells, channels, space_dim), 
                                 dtype=field_values.dtype, device=self.device)
        
        # Internal cells only:
        idx = self.internal_faces_idx

        # Compute Flux
        phi_o = field_values[...,self.owner[idx],:]
        phi_n = field_values[...,self.neighbour[idx],:]
        dphi = phi_o*self.interp_ws[...,idx,:] + phi_n*(1-self.interp_ws[...,idx,:])

        flux = torch.einsum('itjk,ijl->itjkl', dphi, self.Sf[:,idx,:])

        grad_field.index_add_(2, self.owner[idx], flux)
        grad_field.index_add_(2, self.neighbour[idx], -flux)

        return grad_field
    
    def apply_volume_correction(self, grad_field: torch.tensor) -> torch.tensor:
        return grad_field / self.cell_volumes.reshape(1,1,-1,1,1)

    def add_bc_conditions(self, dict):
        # TODO: this needs to be replaced by something more automated or in init file
        self.bc_conditions = dict

    def add_BC_to_flux(self, grad_field: torch.tensor, field_type: str, field_values: torch.tensor=None, order=1) -> torch.tensor:
        assert field_type is not None

        for patch in self.patch_face_keys:
            face_keys_idx = self.patch_face_keys[patch]
    
            bc_values_at_face = get_boundary_flux(grad_field = grad_field, 
                                                  field_values = field_values,
                                                  bc_patch = self.bc_conditions[field_type][patch],
                                                  owner = self.owner[face_keys_idx],
                                                  order = order,
                                                  face_normals_patch = self.face_normals[face_keys_idx])
            
            if bc_values_at_face is not None:
                # bc_values_at_face[...,:self.dim] may break if we do preassure as dim=1
                bc_flux = torch.einsum('itjk,ijl->itjkl', bc_values_at_face[...,:self.dim], self.Sf[...,face_keys_idx,:])
                grad_field.index_add_(2, self.owner[face_keys_idx], bc_flux)
        
        return grad_field
    
    def compute_derivative(self, field: torch.tensor, field_type:str=None, order:int=1) -> torch.tensor:
        
        # reshape into time dim for function universality
        if len(field.shape) == 3:
            field = field.unsqueeze(1)    # (Batch, Time, Nodes, Channel)
        
        grad_field = self.compute_unweighted_gradient(field)
        
        if self.boundaries:
            assert field_type is not None
            grad_field = self.add_BC_to_flux(grad_field, field_type, field, order=order)
        
        grad_field = self.apply_volume_correction(grad_field)

        if field.shape[1] == 1:
            grad_field = grad_field.permute(0,1,2,4,3).squeeze(1).flatten(start_dim=2)
        return grad_field