import torch
import numpy as np

# Loss Functions
def get_loss_fn(name:str):
    loss_type_map = {'MSE':torch.nn.MSELoss(),
                     'log_MSE':log_MSE(),
                     'LPloss':LpLoss()
                     }
    return loss_type_map[name]

class log_MSE(object):
    def __init__(self):
        pass
    def log_mse(self, x, power=2):
        log_resid = torch.log1p(x.pow(power))
        return torch.mean(log_resid)

    def __call__(self, x, y):
        if y is None:
            return self.log_mse(x)
        elif torch.all(y == 0):
            return self.log_mse(x)
        else:
            raise NotImplementedError('Only comparison to zero supported for log MSE')

class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]
        # editted this here
        node_index = np.argmax(x.shape)
        nodes = x.shape[node_index]
        dim_2d = int(np.sqrt(nodes))
        
        #Assume uniform mesh
        h = 1.0 / (dim_2d - 1.0) # this is technically dx or dy

        # lets try this:
        #h = 1

        all_norms = (h**(self.d/self.p))*torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.abs(x, y)
    
# Loss Balancer