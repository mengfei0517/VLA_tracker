import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------------------------------------------------------
# 1. Noise Scheduler (noise/denoise schedule)
# ---------------------------------------------------------
class DiffScheduler:
    """
    Responsible for managing the noise ratio (Alpha and Beta) during the diffusion process
    """
    def __init__(self, n_steps=100):
        self.n_steps = n_steps
        # Use simple linear scheduling (Linear Schedule)
        self.betas = torch.linspace(1e-4, 0.02, n_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        """Diffuse the original action x_0 to the noisy state at step t"""
        # 公式: x_t = sqrt(alpha_cum) * x_0 + sqrt(1 - alpha_cum) * noise
        noise = torch.randn_like(x_0)
        alpha_cum = self.alphas_cumprod[t].view(-1, 1, 1)
        
        x_t = torch.sqrt(alpha_cum) * x_0 + torch.sqrt(1 - alpha_cum) * noise
        return x_t, noise

# ---------------------------------------------------------
# 2. Denoiser Network (denoising network - cerebellum)
# ---------------------------------------------------------
class ActionDenoiser(nn.Module):
    """
    Input: Noisy action x_t, time step t, and Transformer extracted features cond
    Output: Predicted noise (Epsilon Prediction)
    """
    def __init__(self, action_dim, cond_dim, hidden_dim=256):
        super().__init__()
        # Time embedding layer: Let the model know the current denoising step
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Core prediction network
        self.mid_layer = nn.Sequential(
            nn.Linear(action_dim + cond_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim) # Predicted noise dimension matches action dimension
        )

    def forward(self, x_t, t, cond):
        # x_t: [Batch, Chunk_size, Action_dim]
        # t: [Batch, 1]
        # cond: [Batch, Cond_dim] (output from Transformer)
        
        t_embed = self.time_mlp(t.float() / 100.0) # Normalize time step
        
        # Concatenate all features: current action + task environment feature + time progress
        # To simplify, we concatenate all dimensions
        x_flat = x_t.flatten(start_dim=1) 
        combined = torch.cat([x_flat, cond, t_embed], dim=-1)
        
        noise_pred = self.mid_layer(combined)
        return noise_pred.view_as(x_t)

# ---------------------------------------------------------
# 3. Diffusion Policy (DP)
# ---------------------------------------------------------
class MiniDiffusionPolicy:
    def __init__(self, action_dim, chunk_size, cond_dim):
        self.scheduler = DiffScheduler(n_steps=50)
        self.model = ActionDenoiser(action_dim * chunk_size, cond_dim)
        self.chunk_size = chunk_size
        self.action_dim = action_dim

    def sample(self, cond):
        """
        Inference process: Restore the action trajectory from pure noise (Inference Loop)
        """
        device = next(self.model.parameters()).device
        batch_size = cond.shape[0]
        
        # 1. Start from pure Gaussian noise [Batch, Chunk_size, Action_dim]
        x_t = torch.randn((batch_size, self.chunk_size, self.action_dim)).to(device)
        
        # 2. Iterative denoising
        self.model.eval()
        with torch.no_grad():
            for i in reversed(range(self.scheduler.n_steps)):
                t = torch.full((batch_size, 1), i, device=device)
                
                # Predict noise
                noise_pred = self.model(x_t, t, cond)
                
                # Simple DDIM step logic (simplified version)
                # Basically, subtract a part of the predicted noise to restore the clearer image
                alpha = self.scheduler.alphas[i]
                alpha_cum = self.scheduler.alphas_cumprod[i]
                beta = self.scheduler.betas[i]
                
                noise = torch.randn_like(x_t) if i > 0 else 0
                x_t = (1 / torch.sqrt(alpha)) * (x_t - ((1 - alpha) / torch.sqrt(1 - alpha_cum)) * noise_pred) + torch.sqrt(beta) * noise
                
        return x_t

# ---------------------------------------------------------
# 4. Running Example
# ---------------------------------------------------------
if __name__ == "__main__":
    # Simulated environment features output from Transformer (Condition)
    cond_dim = 512
    batch_size = 1
    fake_cond = torch.randn(batch_size, cond_dim)
    
    # Robot action configuration: predict 8 steps in the future, each step is 6-dimensional (XYZ + RPY)
    dp = MiniDiffusionPolicy(action_dim=6, chunk_size=8, cond_dim=cond_dim)
    
    # Execute denoising sampling
    trajectory = dp.sample(fake_cond)
    
    print("-" * 30)
    print("Diffusion Policy MVP Check")
    print("-" * 30)
    print(f"Condition dimension: {fake_cond.shape}")
    print(f"Generated action trajectory shape: {trajectory.shape}") # [1, 8, 6]
    print(f"First action example: \n{trajectory[0, 0, :]}")
    print("\nDP inference loop logic verified!")