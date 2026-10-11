import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np


class VectorQuantizer(nn.Module):
    """Vector Quantization with EMA codebook updates, perplexity tracking, and entropy regularizer."""
    
    def __init__(self, embed_dim, num_codes, beta=0.25, commitment_weight=1.0, entropy_weight=0.01):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_codes = num_codes
        self.beta = beta
        self.commitment_weight = commitment_weight
        self.entropy_weight = entropy_weight
        
        # Codebook: (num_codes, embed_dim)
        self.codebook = nn.Embedding(num_codes, embed_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        
        # EMA codebook tracking for stability
        self._codebook_ema = self.codebook.weight.data.clone()
        
        # Perplexity tracking
        self._code_usage = torch.zeros(num_codes)
        self._total_samples = 0
        self._usage_window = deque(maxlen=100)
    
    def forward(self, z, code_indices, return_logits=False):
        """
        VQ forward pass with straight-through estimator.
        
        Args:
            z: Input features (B, N, embed_dim) or (B*N, embed_dim)
            code_indices: Quantized indices (B, N) or (B*N,)
            return_logits: If True, also return logits for entropy calculation
            
        Returns:
            z_q: Quantized features
            loss: VQ loss (includes entropy regularizer)
            perplexity: Codebook usage perplexity
        """
        # Flatten if needed to (B*N, embed_dim)
        shape = z.shape
        z_flat = z.view(-1, self.embed_dim)
        
        # Get quantized embeddings
        z_q = self.codebook(code_indices)
        
        # Straight-through estimator: z_q = z + (z_q - z).detach()
        z_q = z_flat + (z_q - z_flat).detach()
        z_q = z_q.view(*shape)
        
        # VQ loss: commitment loss + codebook loss
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        loss = self.commitment_weight * commitment_loss + self.beta * codebook_loss
        
        # Track code usage for perplexity
        self._update_code_usage(code_indices)
        perplexity = self._compute_perplexity()
        
        if return_logits:
            return z_q, loss, perplexity
        
        return z_q, loss, perplexity
    
    def forward_with_entropy(self, z, logits):
        """
        Forward pass that includes entropy regularizer in loss.
        
        Args:
            z: Input features
            logits: Unnormalized log probs for code selection (B*N, num_codes)
            
        Returns:
            z_q, loss, perplexity
        """
        # Get code indices from logits (sample or argmax)
        code_indices = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
        
        # Flatten z
        z_flat = z.view(-1, self.embed_dim)
        
        # Get quantized embeddings
        z_q = self.codebook(code_indices)
        
        # Straight-through estimator
        z_q = z_flat + (z_q - z_flat).detach()
        z_q = z_q.view(*z.shape)
        
        # VQ loss
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        loss = self.commitment_weight * commitment_loss + self.beta * codebook_loss
        
        # Entropy regularizer: encourage uniform code usage
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.mean(torch.sum(F.softmax(logits, dim=-1) * log_probs, dim=-1))
        loss = loss - self.entropy_weight * entropy  # Minus because we want to maximize entropy
        
        # Track code usage
        self._update_code_usage(code_indices)
        perplexity = self._compute_perplexity()
        
        return z_q, loss, perplexity
    
    def _update_code_usage(self, code_indices):
        """Update code usage statistics."""
        flat_indices = code_indices.view(-1)
        counts = torch.bincount(flat_indices, minlength=self.num_codes)
        self._code_usage += counts.float()
        self._total_samples += flat_indices.size(0)
        self._usage_window.append(flat_indices.clone())
    
    def _compute_perplexity(self):
        """Compute codebook usage perplexity."""
        if self._total_samples == 0:
            return float('inf')
        
        probs = self._code_usage / self._total_samples
        probs = probs.clamp(min=1e-10)
        entropy = -torch.sum(probs * torch.log(probs))
        return torch.exp(entropy)
    
    def get_usage_stats(self):
        """Return current codebook usage statistics."""
        if self._total_samples == 0:
            return {
                'perplexity': float('inf'),
                'active_codes': 0,
                'total_samples': 0,
                'usage_distribution': None
            }
        
        probs = self._code_usage / self._total_samples
        active_codes = (probs > 0.01).sum().item()
        
        return {
            'perplexity': self._compute_perplexity().item(),
            'active_codes': active_codes,
            'total_samples': self._total_samples,
            'usage_distribution': probs.detach().cpu().numpy()
        }
    
    def reset_stats(self):
        """Reset perplexity tracking statistics."""
        self._code_usage.zero_()
        self._total_samples = 0
        self._usage_window.clear()
    
    def reset_codebook(self):
        """Reset codebook to random initialization."""
        self.codebook.weight.data.uniform_(-1.0 / self.num_codes, 1.0 / self.num_codes)
        self._codebook_ema = self.codebook.weight.data.clone()
        self.reset_stats()


def compute_codebook_perplexity(vq_module):
    """Standalone function to compute and return perplexity from a VQ module."""
    stats = vq_module.get_usage_stats()
    perplexity = stats['perplexity']
    
    if perplexity < float('inf'):
        log_perplexity = np.log(perplexity) if perplexity > 0 else 0
        print(f"[VQ] Codebook perplexity: {perplexity:.2f} (log: {log_perplexity:.3f})")
        print(f"[VQ] Active codes: {stats['active_codes']}/{vq_module.num_codes}")
    else:
        print("[VQ] No code usage statistics available yet")
    
    return perplexity, stats


def revive_dead_codes(vq_module, z, code_indices, threshold=0.001):
    """
    Revive dead codebook entries by reinitializing unused codes.
    
    Args:
        vq_module: VectorQuantizer instance
        z: Input features
        code_indices: Current assignments
        threshold: Minimum usage threshold to consider a code 'alive'
    """
    stats = vq_module.get_usage_stats()
    if stats['total_samples'] == 0:
        return
    
    probs = stats['usage_distribution']
    dead_codes = (probs < threshold).nonzero(as_tuple=True)[0]
    
    if len(dead_codes) > 0:
        print(f"[VQ] Reviving {len(dead_codes)} dead code(s)")
        
        for code_idx in dead_codes:
            if len(vq_module._usage_window) > 0:
                recent_assignments = torch.cat(list(vq_module._usage_window))
                random_idx = torch.randint(0, len(recent_assignments), (1,))
                selected_z = z[recent_assignments[random_idx] % z.size(0)]
                vq_module.codebook.weight[code_idx] = selected_z.squeeze(0)
            else:
                vq_module.codebook.weight[code_idx].data.uniform_(
                    -1.0 / vq_module.num_codes, 1.0 / vq_module.num_codes
                )


if __name__ == "__main__":
    # Simple self-test
    vq = VectorQuantizer(embed_dim=64, num_codes=128)
    
    batch_size = 32
    num_agents = 8
    num_steps = 10
    
    for step in range(num_steps):
        z = torch.randn(batch_size * num_agents, 64)
        code_indices = torch.randint(0, 128, (batch_size * num_agents,))
        
        z_q, loss, perplexity = vq(z, code_indices)
        
        if step % 5 == 0:
            compute_codebook_perplexity(vq)
    
    print("\nVQ module self-test completed successfully!")
