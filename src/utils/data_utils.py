import pyvista as pv

def get_vtk_file_reader(case_name: str):

    # get file extension:
    extension = case_name.split('.')[-1]
    if extension == 'foam':
        return pv.POpenFOAMReader(case_name)
    else:
        raise NotImplementedError
    
def vtk_boundary_finder():
    raise NotImplementedError
    # Idea is to either get the boundary patches through layers:
    # i.e. if the boundaries are in a subkey or whether they are next to
    # internal field patches
    # additionally, we want to be able to parse a flag map from a vtp file (instead)

def get_bc_dict(file=None):

    if file == None:
        print('resorting to default bc_dict, this line needs to be superseeded')
        U_bc_dict = {
            'in':{ "type":'fixedValue', "value":[1,0,0] },
            'out':{ "type":'zeroGradient', },  
            'cylinder':{ "type":'fixedValue', "value":[0,0,0] },
            'sym1':{ "type":'symmetryPlane' },
            'sym2':{ "type":'symmetryPlane' },
            'front':{ "type":'empty' },
            'back':{ "type":'empty' }        
            }
        p_bc_dict = {
            'in':{ "type":'zeroGradient' },
            'out':{ "type":'fixedValue', "value":0 },  
            'cylinder':{ "type":'zeroGradient' } ,
            'sym1':{ "type":'symmetryPlane' },
            'sym2':{ "type":'symmetryPlane' },
            'front':{ "type":'empty' },
            'back':{ "type":'empty' }     
            }
        return {'U':U_bc_dict, 'p':p_bc_dict}
    else:
        raise NotImplementedError

def parse_openfoam_bc_file():
    raise NotImplementedError
    # Idea is to either get the boundary patches through layers:
    # i.e. if the boundaries are in a subkey or whether they are next to
    # internal field patches
    # additionally, we want to be able to parse a flag map from a vtp file (instead)


