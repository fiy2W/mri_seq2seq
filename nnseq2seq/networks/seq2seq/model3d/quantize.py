import numpy as np
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftVectorQuantizer(nn.Module):
    def __init__(
        self,
        n_e,
        e_dim,
        entropy_loss_ratio=0.01,
        tau=0.07,
        l2_norm=True,
        show_usage=False,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.tau = tau
        
        # Single embedding layer for all codebooks
        self.embedding = nn.Parameter(torch.randn(n_e, e_dim))
        self.embedding.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        
        if self.l2_norm:
            self.embedding.data = F.normalize(self.embedding.data, p=2, dim=-1)
        
        if self.show_usage:
            self.register_buffer("codebook_used", torch.zeros(65536))

    def forward(self, z):
        # Handle different input shapes
        if z.dim() == 5:
            z = torch.einsum('b c h w d -> b h w d c', z).contiguous()
        
        batch_size, h, w, d, _ = z.shape
        
        # Apply L2 norm if needed
        embedding = F.normalize(self.embedding, p=2, dim=-1) if self.l2_norm else self.embedding
        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            
        z_flat = z.view(-1, self.e_dim)
        
        logits = torch.einsum('be, ke -> bk', z_flat, embedding.detach())
        
        # Calculate probabilities
        probs = F.softmax(logits / self.tau, dim=-1)
        indices = torch.argmax(probs, dim=-1)
        
        # Quantize
        z_q_soft = torch.einsum('bk, ke -> be', probs, embedding)
        z_q_hard = torch.einsum('bk, ke -> be', F.one_hot(indices, self.n_e).float(), embedding)
        
        # Reshape back
        z_q_soft = z_q_soft.view(batch_size, h, w, d, self.e_dim).contiguous()
        z_q_hard = z_q_hard.view(batch_size, h, w, d, self.e_dim).contiguous()
        
        # Calculate losses if training
        vq_loss = self.entropy_loss_ratio * compute_entropy_loss(logits.view(-1, self.n_e))
        
        # Reshape back to match original input shape
        if len(z.shape) == 5:
            z_q_soft = torch.einsum('b h w d c -> b c h w d', z_q_soft)
            z_q_hard = torch.einsum('b h w d c -> b c h w d', z_q_hard)
        
        indices = indices.view(batch_size, h, w, d)
        
        return z_q_soft, vq_loss, (z_q_hard, indices)
    
    @torch.no_grad()
    def get_codebook_entry(self, indices):
        b, h, w, d = indices.shape
        indices = indices.view(-1)
        embedding = F.normalize(self.embedding, p=2, dim=-1) if self.l2_norm else self.embedding
        z_q_hard = torch.einsum('bk, ke -> be', F.one_hot(indices, self.n_e).float(), embedding.detach())
        z_q_hard = z_q_hard.view(b, h, w, d, self.e_dim).contiguous()
        z_q_hard = torch.einsum('b h w d c -> b c h w d', z_q_hard)
        return z_q_hard


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-6))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss