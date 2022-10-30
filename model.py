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

def render_path(model, imgs, poses, bds, focal, N_samples, N_importance, chunk, perturb, N_rand, network_fov, use_viewdirs, use_disp, use_noise, use_ndc):
    # Get batch size
    batch_size = poses.shape[0]
    
    # Get image size
    H = imgs.shape[2]
    W = imgs.shape[3]
    
    # Get focal length
    focal = focal[:, 0, 0]
    
    # Get ray directions
    rays_o, rays_d = get_rays(H, W, focal, poses, network_fov, use_ndc)
    
    # Get ray batch indices
    rays_o = rays_o.repeat(batch_size, 1, 1)
    rays_d = rays_d.repeat(batch_size, 1, 1)
    rays_b = torch.arange(batch_size, dtype=torch.long).view(-1, 1, 1).repeat(1, H, W).view(-1)
    
    # Get ray origins and directions
    rays_o = rays_o.view(-1, 3)
    rays_d = rays_d.view(-1, 3)
    
    # Get ray bounds
    near, far = get_bounds(bds, poses, rays_o, rays_d, use_ndc)
    
    # Sample along ray
    z_vals = torch.linspace(0.0, 1.0, N_samples, device=imgs.device)
    z_vals = near[..., None] * (1.0 - z_vals) + far[..., None] * z_vals
    z_vals = z_vals.view(batch_size, -1)
    
    # Get ray directions
    if use_viewdirs:
        viewdirs = rays_d.view(batch_size, H, W, 3)
        viewdirs = viewdirs[:, None, None, :, :].repeat(1, N_samples, N_importance, 1, 1).view(batch_size, -1, 3)
    else:
        viewdirs = None
    
    # Render
    rgb, disp, acc, extras = render(model, rays_o, rays_d, z_vals, rays_b, chunk, perturb, N_rand, use_viewdirs, use_disp, viewdirs)
    
    # Reshape
    rgb = rgb.view(batch_size, H, W, 3)
    disp = disp.view(batch_size, H, W)
    acc = acc.view(batch_size, H, W)
    extras = extras.view(batch_size, H, W)
    
    return rgb, disp, acc, extras

def render(model, rays_o, rays_d, z_vals, rays_b, chunk, perturb, N_rand, use_viewdirs, use_disp, viewdirs):
    # Get batch size
    batch_size = rays_o.shape[0]
    
    # Get number of rays
    N_rays = rays_o.shape[0]
    
    # Get number of samples
    N_samples = z_vals.shape[1]
    
    # Get number of chunks
    N_chunks = max(N_rays // chunk, 1)
    
    # Initialize rgb, disp, acc, extras
    rgb = torch.zeros(N_rays, 3, device=rays_o.device)
    disp = torch.zeros(N_rays, device=rays_o.device)
    acc = torch.zeros(N_rays, device=rays_o.device)
    extras = torch.zeros(N_rays, device=rays_o.device)
    
    # Render
    for i in range(N_chunks):
        # Get chunk indices
        chunk_idx = torch.arange(i * chunk, min((i + 1) * chunk, N_rays), device=rays_o.device)
        
        # Get chunk rays
        chunk_rays_o = rays_o[chunk_idx]










def test():
    print("Testing!")
if __name__ == "__main__":
    test()