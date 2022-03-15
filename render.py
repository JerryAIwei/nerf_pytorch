from pickle import NONE
from tkinter.messagebox import NO
from setup import *
from sampling import *
from encoder import *

def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, rand=False):
    H = rays_o.shape[0]
    z_vals = torch.linspace(near, far, N_samples).to(device) 
    pts_flat = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
    pts_flat = encoder_position(pts_flat, 4)
    raw = network_fn(pts_flat)
    sigma = F.relu(raw[...,0].reshape(H, -1))
    color = torch.sigmoid(raw[...,1:].reshape(H, -1, 3))
    dists = torch.concat([z_vals[..., 1:] - z_vals[..., :-1], torch.broadcast_to(torch.from_numpy(np.array([1e10])), z_vals[...,:1].shape).to(device)], -1)
    alpha = 1.-torch.exp(-sigma * dists)  
    weights = alpha * torch.cumprod(1.-alpha + 1e-10, -1)
    rgb_map = torch.sum(weights[...,None] * color, -2)
    acc_map = torch.sum(weights, -1)
    rgb_map = rgb_map + (1. - acc_map[..., None])
#Todo:
# test pipeline
# view direction
# random sampling
# weight sampling
    return rgb_map, weights