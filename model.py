import numpy as np 
import torch 

# ((x,y,z), theta, phi) ---> ((r,g,b), sigma(density))

class NeRF(torch.nn.Module):
    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connections=4,
        num_encoding_func_xyz=6,
        num_encoding_func_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        ):
            super(NeRF, self).__init__()
            if include_input_xyz:
                include_input_xyz = 3
            else:
                include_input_xyz = 0
            self.dim_xyz = include_input_xyz+2*3*num_encoding_func_xyz
            self.dim_dir = include_input_dir+2*3*num_encoding_func_dir

            self.layers_xyz = torch.nn.ModuleList()
            self.use_viewdirs = use_viewdirs
            self.layers_xyz.append(torch.nn.Linear(self.dim_xyz+256, 256))
            #iterate through layers
            for i in range(1,8): # traverse through 8 layers
                if i == 4: # if idx = 4
                    self.layers_xyz.append(torch.nn.Linear(self.dim_xyz+256, 256))
                else:
                    self.layers_xyz.append(torch.nn.Linear(256,256))
            self.fc
        
        pass 
    pass

def test():
    print("Testing!")
if __name__ == "__main__":
    test()