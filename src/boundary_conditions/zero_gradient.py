import torch
import numpy as np

def zero_gradient_bc(field_values: torch.tensor, 
                   owner: torch.tensor,
                   order = 1,
                   **kwargs) -> torch.tensor:
    
    if order > 1:
        return
    else: 
        return field_values[...,owner,:]