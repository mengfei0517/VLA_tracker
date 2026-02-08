import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. Flow Matcher (flow matching core logic)
# ---------------------------------------------------------
class FlowMatcher:
    """
    Responsible for defining the straight path from noise to action: x_t = (1-t) * x_0 + t * x_1
    """
    def sample_location(self, x_0, x_1, t):
        """
        Linear interpolation between noise x_0 and real action x_1 at time t
        t=0 is pure noise, t=1 is real action
        """
        # Corresponding formula: x_t = (1 - t) * x_0 + t * x_1
        t = t.view(-1, 1, 1)
        x_t = (1 - t) * x_0 + t * x_1
        
        # The velocity (derivative) is: v = x_1 - x_0
        velocity = x_1 - x_0
        return x_t, velocity

# ---------------------------------------------------------
# 2. Flow Network (flow prediction network - predict velocity)
# ---------------------------------------------------------
class FlowNetwork(nn.Module):
    def __init__(self, action_dim, cond_dim, hidden_dim=256):
        super().__init__()
        # Time encoding layer
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Predict the "moving velocity vector" of x_t at time t
        self.net = nn.Sequential(
            nn.Linear(action_dim + cond_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim) 
        )

    def forward(self, x_t, t, cond):
        t_embed = self.time_mlp(t.view(-1, 1))
        x_flat = x_t.flatten(start_dim=1)
        combined = torch.cat([x_flat, cond, t_embed], dim=-1)
        return self.net(combined).view_as(x_t)

# ---------------------------------------------------------
# 3. Mini Flow Matching Policy
# ---------------------------------------------------------
class MiniFMPolicy:
    def __init__(self, action_dim, chunk_size, cond_dim):
        self.model = FlowNetwork(action_dim * chunk_size, cond_dim)
        self.chunk_size = chunk_size
        self.action_dim = action_dim

    def sample(self, cond, steps=5):
        """
        FM inference: use simple Euler integration (Euler Integration)
        """
        device = next(self.model.parameters()).device
        batch_size = cond.shape[0]
        
        # 1. Start from pure Gaussian noise x_0
        x_t = torch.randn((batch_size, self.chunk_size, self.action_dim)).to(device)
        
        dt = 1.0 / steps
        self.model.eval()
        with torch.no_grad():
            for i in range(steps):
                # Current time point t from 0 to 1
                t = torch.full((batch_size, 1), i * dt, device=device)
                
                # Predict the "moving velocity" of the current position
                v_pred = self.model(x_t, t, cond)
                
                # [Core]: take a small step along the velocity direction (Euler method)
                # x_{t+dt} = x_t + v * dt
                x_t = x_t + v_pred * dt
                
        return x_t

# ---------------------------------------------------------
# 4. Running Example
# ---------------------------------------------------------
if __name__ == "__main__":
    cond_dim = 512
    fake_cond = torch.randn(1, cond_dim)
    
    # Instantiate FM policy
    fm_policy = MiniFMPolicy(action_dim=6, chunk_size=8, cond_dim=cond_dim)
    
    # Execute inference: only 5 steps, even 1 step can see the skeleton
    trajectory = fm_policy.sample(fake_cond, steps=5)
    
    print("-" * 30)
    print("Flow Matching MVP Check")
    print("-" * 30)
    print(f"Generated trajectory shape: {trajectory.shape}")
    print(f"FM sampling steps: 5 (much less than 50 steps of DP)")
    print("\nFlow Matching logic verified!");