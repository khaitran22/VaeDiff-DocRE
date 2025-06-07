import numpy as np
from losses import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from dist import Normal
from improved_diffusion.nn import timestep_embedding

def reparameterization(dim_z, mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = torch.randn(size=(mu.size(0), dim_z)).to(mu.device)
    z = sampled_z * std + mu
    return z

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.gamma
    
class VAE(nn.Module):
    def __init__(self, args, config):
        super(VAE, self).__init__()

        self.args = args
        self.dim_h = config.hidden_size
        self.dim_emb = config.hidden_size // 4
        self.dim_z = args.latent_size
        
        self.encoder = nn.Sequential(
            RMSNorm(self.dim_h),
            nn.Linear(self.dim_h, self.dim_emb),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.dim_emb, self.dim_emb),
            RMSNorm(self.dim_emb),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.dim_emb, self.dim_z),
        )
        self.mu = nn.Linear(self.dim_z, self.dim_z)
        self.log_sigma = nn.Linear(self.dim_z, self.dim_z)

        self.decoder = nn.Sequential(
            nn.Linear(self.dim_z, self.dim_emb),
            nn.LeakyReLU(0.2, inplace=True),
            RMSNorm(self.dim_emb),
            nn.Linear(self.dim_emb, self.dim_emb),
            nn.LeakyReLU(0.2, inplace=True),
            RMSNorm(self.dim_emb),
            nn.Linear(self.dim_emb, self.dim_h * 2)
        )

    def encode(self, features):
        z = self.encoder(features)
        mu = self.mu(z)
        log_sigma = self.log_sigma(z)
        return mu, log_sigma
    
    def decode(self, z):
        logits = self.decoder(z)
        return logits
    
    def decoder_output(self, logits):
        mu, log_sigma = torch.chunk(logits, 2, dim=1)
        return Normal(mu, log_sigma)
    
    def reconstruction_loss(self, output, input):
        log_p_input = output.log_p(input)
        return (-1) * log_p_input.sum()
    
    def forward(self, features):
        p_mu, p_log_sigma = self.encode(features)
        p_dist = Normal(p_mu, p_log_sigma)
        all_p_latent, _ = p_dist.sample()
        
        all_log_p = p_dist.log_p(all_p_latent)
        logits = self.decode(all_p_latent)
        
        return logits, all_p_latent, all_log_p