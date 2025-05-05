import vtk

def get_vtk_file_reader(case_name: str):

    # get file extension:
    extension = case_name.split('.')[-1]

    if extension == '.foam':
        return vtk.vtkPOpenFOAMReader()
    else:
        raise NotImplementedError