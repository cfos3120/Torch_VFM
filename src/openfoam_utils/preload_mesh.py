import numpy as np
import torch
from types import SimpleNamespace
from pathlib import Path
from .polymesh_parsing import *

'''
This is a quick meshing option if referencing the OpenFOAM polymesh data and mesh 
information has been pre-written using OpenFOAM commands
'''
class preprocessed_OpenFOAM_mesh(object):
    def __init__(self, case_dir, dim=3, device='cpu', dtype=torch.float32, bc_dict=None):

        # Assert all files are there
        assert case_dir[-5:] == '.foam'
        case_path = Path(case_dir).parent
        assert Path(f"{case_path}/0/Vc").exists() 
        assert Path(f"{case_path}/constant/weights").exists()
        assert Path(f"{case_path}/constant/polyMesh/Cc").exists()
        assert Path(f"{case_path}/constant/polyMesh/owner").exists()
        assert Path(f"{case_path}/constant/polyMesh/neighbour").exists()
        assert Path(f"{case_path}/constant/polyMesh/delta").exists()
        assert Path(f"{case_path}/constant/polyMesh/Sf").exists()
        assert Path(f"{case_path}/constant/polyMesh/boundary").exists()

        # Load in pre-processed mesh information
        self.bc_conditions = bc_dict
        self.mesh = SimpleNamespace()
        self.mesh.dim = dim
        
        self.mesh.cell_centers = torch.tensor(parse_openfoam_face_values(f"{case_path}/constant/polyMesh/Cc", internal_only=True), dtype=dtype)
        self.mesh.n_cells = self.mesh.cell_centers.shape[0]

        if self.mesh.dim == 2:
            z_len = read_vertices(f"{case_path}/constant/polyMesh/points")[...,-1]
            z_len = z_len.max() - z_len.min()
            assert self.mesh.cell_centers[...,-1].max() == self.mesh.cell_centers[...,-1].min()
        else: 
            z_len = 1

        self.mesh.cell_centers = self.mesh.cell_centers[...,:dim]
        self.mesh.face_owners = torch.tensor(parse_owner_file(f"{case_path}/constant/polyMesh/owner"), dtype=torch.int64)
        self.mesh.face_neighbors = torch.tensor(parse_owner_file(f"{case_path}/constant/polyMesh/neighbour"), dtype=torch.int64)
        self.mesh.internal_faces = torch.arange(len(self.mesh.face_neighbors))
        self.mesh.num_internal_faces = len(self.mesh.face_neighbors)
        self.mesh.cell_volumes = torch.tensor(load_openfoam_scalar_field(f"{case_path}/0/Vc"), dtype=dtype)/z_len

        bc_dict = read_boundary_dict(f"{case_path}/constant/polyMesh/boundary")
        Sf_raw = torch.tensor(parse_openfoam_face_values(f"{case_path}/constant/polyMesh/Sf"), dtype=dtype)[...,:dim]
        delta_raw = torch.tensor(parse_openfoam_face_values(f"{case_path}/constant/polyMesh/delta"), dtype=dtype)[...,:dim]
        self.internal_face_weights = torch.tensor(parse_openfoam_face_values(f"{case_path}/constant/weights", field_type='auto', internal_only=False), dtype=dtype)
        
        # read in Internal face + non-empty Boundary face values
        self.mesh.face_areas = torch.empty_like(self.mesh.face_owners, dtype=dtype).reshape(-1,1).repeat(1,dim)
        self.mesh.cell_center_vectors = torch.empty_like(self.mesh.face_owners, dtype=dtype).reshape(-1,1).repeat(1,dim)
        
        # allocate them (avoid empty faces)
        self.mesh.patch_face_keys = {}
        for i, (key, value) in enumerate(bc_dict.items()):
            startFace = value['startFace']
            nFaces = value['nFaces']
            if i == 0:
                self.mesh.face_areas[:startFace,:] = Sf_raw[:startFace,:dim]/z_len
                self.mesh.cell_center_vectors[:startFace,:] = delta_raw[:startFace,:dim]
                last_face = startFace
            
            self.mesh.patch_face_keys[key] = torch.arange(startFace,startFace+nFaces, dtype=torch.int64)
            if value['type'] != 'empty':
                self.mesh.face_areas[startFace:startFace+nFaces,:] = Sf_raw[last_face:last_face+nFaces,:dim]/z_len
                self.mesh.cell_center_vectors[startFace:startFace+nFaces,:] = delta_raw[last_face:last_face+nFaces,:dim]
                last_face += nFaces

        self.mesh.face_areas_mag = torch.norm(self.mesh.face_areas, dim=-1, keepdim=False).to(dtype) 
        

        # Orthogonality
        self.correction_method = None
        self.delta = self.mesh.face_areas/self.mesh.face_areas_mag.unsqueeze(-1)
        self.delta_mag = torch.norm(self.delta, dim=-1, keepdim=False).to(dtype)
        self.d = self.mesh.cell_center_vectors.to(dtype)
        self.d_mag = torch.norm(self.d, dim=-1, keepdim=False).to(dtype)

        # send all to device
        self.device = torch.device(device)
        self.dtype = dtype
        self.to(device)

    def to(self,device):
        for attr_name, attr_value in vars(self.mesh).items():
            if isinstance(attr_value, torch.Tensor):
                setattr(self.mesh, attr_name, attr_value.to(device))

        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, torch.Tensor):
                setattr(self, attr_name, attr_value.to(device))