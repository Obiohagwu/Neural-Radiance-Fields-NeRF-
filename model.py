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
            self.fc_feat = torch.nn.Linear(256, 256)
            self.fc_alpha = torch.nn.Linear(256, 1)

            self.layers_dir = torch.nn.ModuleList()
            self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir, 128))
            for i in range(3):
                self.layers_dir.append(torch.nn.Linear(128, 128))
            self.fc_rgb = torch.nn.Linear(128, 3)
            self.relu = torch.nn.functional.relu
        
    def forward(self, x):
            xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
            for i in range(8):
                if i == 4:
                    x = self.layers_xyz[i](torch.cat((xyz, x), -1))
                else:
                    x = self.layers_xyz[i](x)
                x = self.relu(x)
            feat = self.fc_feat(x)
            alpha = self.fc_alpha(feat)
            if self.use_viewdirs:
                x = self.layers_dir[0](torch.cat((feat, dirs), -1))
            else:
                x = self.layers_dir[0](feat)
            x = self.relu(x)
            for i in range(1, 3):
                x = self.layers_dir[i](x)
                x = self.relu(x)
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
       

def test():
    print("Testing!")
if __name__ == "__main__":
    test()