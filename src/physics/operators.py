import torch

'''
Time Derivative
Takes in a range of prior solutions and calculates the change in solution over time.
Available methods are steady-state(returns zero), Euler (first order) and Backward (second order)
'''
class Temporal_Differentiator():

    @staticmethod
    def caclulate(solution:torch.Tensor, time_step, input_solution:torch.Tensor=None, method='backwards'):
        assert len(solution.shape) == 4
        output_solution_shape = solution.shape
        if input_solution is not None:
            assert len(input_solution.shape) == 4
            solution = torch.cat([input_solution,solution], dim=1)
            input_time_steps = input_solution.shape[1]
        else:
            input_time_steps = 0
        assert solution.shape[1] > 1, 'Must include more than one time-step'

        if method == 'steady':
            return 0
        elif method == 'backwards':
            dt_out =  Temporal_Differentiator.backward_method(solution, time_step, starting_idx=input_time_steps)
        elif method == 'euler':
            dt_out =  Temporal_Differentiator.euler_method(solution, time_step, starting_idx=input_time_steps)
        else:
            raise NotImplementedError(f'Time Derivation Method "{method}" not supported, please select "steady","euler" or "backwards"')

        assert dt_out.shape == output_solution_shape, 'May need to fix something here if an assertion error is thrown'
        
        # pad to 3D from 2D if not a scalar
        if solution.shape[-1] == 2:
            dt_out = torch.nn.functional.pad(dt_out,(0,1))
        return dt_out

    @staticmethod
    def euler_method(solution:torch.Tensor, time_step, starting_idx=0):
        if starting_idx != 0:
            starting_idx -= 1
        dt_out = (solution[:,starting_idx+1:,...] - solution[:,starting_idx:-1,...])/time_step
        return dt_out
        
    @staticmethod
    def backward_method(solution:torch.Tensor, time_step, starting_idx=0):
        if starting_idx != 0:
            starting_idx -= 2
        if starting_idx == -1:
            starting_idx +=1
            solution = torch.cat([solution[:,[0],...],solution], dim=1)
        dt_out = ((3/2)*solution[:,starting_idx+2:,...] - 2*solution[:,starting_idx+1:-1,...] + (1/2)*solution[:,starting_idx:-2,...])/time_step
        return dt_out

'''
Face Interpolater
Linear interpolates the values from cell centres to face centres
using specified weights according to distance to the face. 
'''
def interpolate_to_faces(self, field: torch.Tensor) -> torch.Tensor:
        
    # Get field shape
    assert len(field.shape) == 4
    batch_size = field.shape[0]
    time_size = field.shape[1]
    channel_size = field.shape[-1]
    
    # Initialize face values
    face_values = torch.zeros((batch_size, time_size, self.mesh.num_internal_faces, channel_size), device=field.device)
        
    # Interpolate for internal faces
    idx = self.mesh.internal_faces
    face_values = field[:,:,self.mesh.face_owners[idx],...]*(self.internal_face_weights).reshape(1,1,-1,1)  + \
                                field[:,:,self.mesh.face_neighbors[idx],...]*(1-self.internal_face_weights).reshape(1,1,-1,1)
    return face_values

'''
Divergence Operator:
Takes the Gauss-Green Object and its mesh
Returns the Divergence calculated with advecting of the field i.e Nabla dot (UU),
Also returns the Gradient field to save computation.
'''
class Divergence_Operator():
    
    @staticmethod
    def caclulate(self, field: torch.Tensor, field_type:str = 'U') -> torch.Tensor:
        
        if len(field.shape) == 3 and field_type == 'p':
            field.unsqueeze(-1)
        batch_size = field.shape[0]
        time_size = field.shape[1]
        channel_size = field.shape[-1]

        div_field = torch.zeros((batch_size, time_size, self.mesh.n_cells, channel_size), dtype=field.dtype, device=self.device)
        grad_field = torch.zeros((batch_size, time_size, self.mesh.n_cells, self.mesh.dim, channel_size), dtype=field.dtype, device=self.device)
        
        div_field, grad_field = Divergence_Operator.internal_flux(self, div_field, grad_field, field)
        div_field, grad_field = Divergence_Operator.boundary_flux(self, div_field, grad_field, field, field_type)
        div_field/= self.mesh.cell_volumes.reshape(1,1,-1,1)
        grad_field/= self.mesh.cell_volumes.reshape(1,1,-1,1,1)
        return div_field, grad_field.flatten(start_dim=-2)

    def internal_flux(self, div_field:torch.tensor, grad_field:torch.tensor, field:torch.tensor) -> torch.Tensor:
        face_values = interpolate_to_faces(self, field)
        idx = self.mesh.internal_faces
        divergence = torch.einsum('btfd,fd->btf', face_values, self.mesh.face_areas[idx]).unsqueeze(-1) * face_values
        gradient = torch.einsum('btfd,fe->btfed', face_values, self.mesh.face_areas[idx])
        div_field.index_add_(2, self.mesh.face_owners[idx], divergence)
        div_field.index_add_(2, self.mesh.face_neighbors[idx], -divergence)
        grad_field.index_add_(2, self.mesh.face_owners[idx], gradient)
        grad_field.index_add_(2, self.mesh.face_neighbors[idx], -gradient)
        return div_field, grad_field
    
    @staticmethod
    def boundary_flux(self, div_field:torch.tensor, grad_field:torch.tensor, field:torch.tensor, field_type:str='U') -> torch.Tensor:
        batch_size = field.shape[0]
        time_size = field.shape[1]
        channel_size = field.shape[-1]

        for patch_name, patch_faces in self.mesh.patch_face_keys.items():
            patch_type = self.bc_conditions[field_type][patch_name]['type']
            
            if patch_type in ('empty','noSlip'):
                continue
            if patch_type == 'fixedValue':
                field_value = torch.tensor(self.bc_conditions[field_type][patch_name]['value'], dtype=self.dtype, device=self.device)
                face_values = field_value.reshape(1, 1, 1, -1).repeat(batch_size, time_size, len(patch_faces), 1)
            elif patch_type == 'symmetryPlane':
                face_values = field[...,self.mesh.face_owners[patch_faces],:] - 2*(torch.einsum('btfc,fc->btf', 
                                                                                                field[...,self.mesh.face_owners[patch_faces],:], 
                                                                                                self.mesh.face_normal_unit_vectors[patch_faces,:]
                                                                                                )).unsqueeze(-1) * self.mesh.face_normal_unit_vectors[patch_faces,:].unsqueeze(0)
            elif patch_type == 'zeroGradient':
                face_values = field[...,self.mesh.face_owners[patch_faces],:]
            else:
                raise NotImplementedError(f'patch type {patch_type} not implemented')
            
            divergence = torch.einsum('btfd,fd->btf', face_values, self.mesh.face_areas[patch_faces]).unsqueeze(-1) * face_values
            gradient = torch.einsum('btfd,fe->btfed', face_values, self.mesh.face_areas[patch_faces])
            
            if patch_type == 'symmetryPlane' and channel_size ==1:  # i.e. Scalar
                gradient = torch.zeros(batch_size, time_size, len(patch_faces), self.mesh.dim, 1, dtype=self.dtype, device=grad_field.device)
                divergence = torch.zeros(batch_size, time_size, len(patch_faces), 1, dtype=self.dtype, device=div_field.device)

            div_field.index_add_(2, self.mesh.face_owners[patch_faces], divergence)
            grad_field.index_add_(2, self.mesh.face_owners[patch_faces], gradient)
    
        return div_field, grad_field

'''
Laplacian Operator:
Takes the Gauss-Green Object and its mesh + supplementary vectors (associated with orthogonality)
Returns the Laplacian calculated with specified implicitly/explicitly and/or non-orthogonal corrections
'''
class Laplacian_Operator():

    @staticmethod
    def caclulate(self, field:torch.tensor, field_type:str = 'U', correction_method:str=None, gradient_field:torch.tensor=None) -> torch.tensor:
        
        # initialize laplacian field
        batch_size = field.shape[0]
        time_size = field.shape[1]

        lap_field = torch.zeros((batch_size, time_size, self.mesh.n_cells, self.mesh.dim), dtype=field.dtype, device=self.device)

        lap_field = Laplacian_Operator.internal_flux(self, lap_field, field, gradient_field=gradient_field)
        lap_field = Laplacian_Operator.boundary_flux(self, lap_field, field, field_type)
        lap_field/= self.mesh.cell_volumes.reshape(1,1,-1,1)

        if correction_method is not None:
            assert gradient_field is not None
            correction_field = torch.zeros((batch_size, time_size, self.mesh.n_cells, self.mesh.dim), dtype=field.dtype, device=self.device)
            correction_field = Laplacian_Operator.internal_flux(self, correction_field, field = field, gradient_field=gradient_field, orthogonal=False, implicit=False)
            #correction_field = Laplacian_Operator.boundary_flux(self, correction_field, field, field_type, orth_vector='k')
            correction_field/= self.mesh.cell_volumes.reshape(1,1,-1,1)
            lap_field += correction_field
        
        return lap_field
    
    @staticmethod
    def internal_flux(self, lap_field:torch.tensor, field:torch.tensor, implicit:bool=True, orthogonal:bool=True, gradient_field:torch.tensor=None) -> torch.tensor:
        idx = self.mesh.internal_faces

        # Orthogonal
        if orthogonal:
            orth_vector = self.delta[idx]
            orth_vector_mag = self.delta_mag[idx]
        else:
            orth_vector = self.k_vector[idx]
            orth_vector_mag = self.k_vector_mag[idx]
        
        if implicit:
            face_gradients = field[:,:,self.mesh.face_neighbors[idx], :] - field[:,:,self.mesh.face_owners[idx],:]
            diffusion = face_gradients * (orth_vector_mag * self.mesh.face_areas_mag[idx]/self.d_mag[idx]).reshape(1, 1, -1, 1)
        else:
            assert gradient_field is not None
            face_gradients = interpolate_to_faces(self, gradient_field).unflatten(dim=-1, sizes=(self.mesh.dim, self.mesh.dim))
            diffusion = torch.einsum('fd, btfde-> btfe',orth_vector, face_gradients)*self.mesh.face_areas_mag[idx].reshape(1,1,-1,1)

        lap_field.index_add_(2, self.mesh.face_owners[idx], diffusion)
        lap_field.index_add_(2, self.mesh.face_neighbors[idx], -diffusion)

        return lap_field
    
    @staticmethod
    def boundary_flux(self, lap_field:torch.tensor, field:torch.tensor, field_type:str='U', orth_vector:str=None) -> torch.tensor:
        for patch_name, patch_faces in self.mesh.patch_face_keys.items():
            patch_type = self.bc_conditions[field_type][patch_name]['type']
            
            if patch_type in ('empty', 'symmetryPlane', 'zeroGradient'):
                continue
            else:
                if orth_vector is None:
                    orth_coefficient = (self.mesh.face_areas_mag[patch_faces]/self.d_mag[patch_faces])
                elif orth_vector == 'Delta':
                    orth_coefficient = torch.einsum('fd,fd -> f', self.mesh.face_normal_unit_vectors[patch_faces], self.delta[patch_faces])
                    orth_coefficient *= self.mesh.face_areas_mag[patch_faces]/(self.delta_mag[patch_faces]**2)
                elif orth_vector == 'k':
                    orth_coefficient = torch.einsum('fd,fd -> f', self.mesh.face_normal_unit_vectors[patch_faces], self.k_vector[patch_faces])
                    k_coefficent = self.mesh.face_areas_mag[patch_faces]/(self.k_vector_mag[patch_faces]**2)
                    orth_coefficient *= k_coefficent
                else:
                    raise KeyboardInterrupt(f'Orthogonal Vector string must be either "Delta", "k" or None. {orth_vector} is not accepted')

            # correct if the vector has zero length
            orth_coefficient[torch.isnan(orth_coefficient)] = 0

            if patch_type == 'fixedValue':
                field_value = field[:,:,self.mesh.face_owners[patch_faces],:]
                bc_value = torch.tensor(self.bc_conditions[field_type][patch_name]['value'], dtype=self.dtype, device=self.device)
                diffusion = (bc_value.reshape(1,1,1,-1)-field_value)*orth_coefficient.reshape(1,1,-1,1)

            elif patch_type == 'noSlip':
                field_value = field[:,:,self.mesh.face_owners[patch_faces],:]
                diffusion = (-field_value)*orth_coefficient.reshape(1,1,-1,1)

            else:
                raise NotImplementedError(f'patch type {patch_type} not implemented')
            
            lap_field.index_add_(2, self.mesh.face_owners[patch_faces], diffusion)

        return lap_field