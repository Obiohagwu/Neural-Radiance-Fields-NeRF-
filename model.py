import numpy as np 
import torch 

# ((x,y,z), theta, phi) ---> ((r,g,b), sigma(density))

class MLP(nn.Module):
    def __init__(self, network_depth, network_width, network_skip, use_viewdirs, use_disp, use_noise):
        super(MLP, self).__init__()
        
        self.use_viewdirs = use_viewdirs
        self.use_disp = use_disp
        self.use_noise = use_noise
        
        # Set network
        self.network = nn.Sequential()
        self.network.add_module('layer_0', nn.Linear(3 + 3 * use_viewdirs, network_width))
        self.network.add_module('activation_0', nn.ReLU())
        
        for i in range(1, network_depth):
            self.network.add_module('layer_{}'.format(i), nn.Linear(network_width + network_skip * i, network_width))
            self.network.add_module('activation_{}'.format(i), nn.ReLU())
        
        self.network.add_module('layer_{}'.format(network_depth), nn.Linear(network_width + network_skip * network_depth, 4 + use_disp))
        
        # Set noise
        if self.use_noise:
            self.noise = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        if self.use_noise:
            x = x + self.noise * torch.randn_like(x)
        
        return self.network(x)


class NeRF(nn.Module):
    def __init__(self, network_depth, network_width, network_skip, use_viewdirs, use_disp, use_noise):
        super(NeRF, self).__init__()
        
        self.use_viewdirs = use_viewdirs
        self.use_disp = use_disp
        self.use_noise = use_noise
        
        # Set network
        self.network = MLP(network_depth, network_width, network_skip, use_viewdirs, use_disp, use_noise)
        
        # Set noise
        if self.use_noise:
            self.noise = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, viewdirs):
        if self.use_noise:
            x = x + self.noise * torch.randn_like(x)
        
        if self.use_viewdirs:
            x = torch.cat([x, viewdirs], -1)
        
        return self.network(x)

    def get_rays(self, H, W, focal, c2w):
        i, j = torch.meshgrid(torch.arange(H), torch.arange(W))
        i = i.float()
        j = j.float()
        dirs = torch.stack([(i - H * 0.5) / focal, -(j - W * 0.5) / focal, -torch.ones_like(i)], -1)
        rays_d = dirs.view(H * W, 3)
        rays_o = torch.zeros_like(rays_d)
        rays_d = normalize_rays(rays_d, c2w)
        rays_o = normalize_rays(rays_o, c2w)
        return rays_o, rays_d

    def render_rays(self, rays_o, rays_d, near, far, N_samples, rand=True, retraw=False, white_bkgd=False):
        batch_size = rays_o.shape[0]
        z_vals = torch.linspace(near, far, N_samples)
        z_vals = z_vals.view(1, 1, N_samples).repeat(batch_size, 1, 1)
        
        if rand:
            z_vals = z_vals + torch.rand_like(z_vals) * (z_vals[:, 1:] - z_vals[:, :-1])
        
        z_vals = z_vals.view(batch_size, -1)
        rays_o = rays_o[:, None, :].repeat(1, N_samples, 1)
        rays_d = rays_d[:, None, :].repeat(1, N_samples, 1)
        rays_o = rays_o.view(batch_size, -1, 3)
        rays_d = rays_d.view(batch_size, -1, 3)









def test():
    print("Testing!")
if __name__ == "__main__":
    test()