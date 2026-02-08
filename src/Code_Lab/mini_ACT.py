import torch
import torch.nn as nn

# ---------------------------------------------------------
# 1. CVAE Encoder (ACT's unique feature: compressing action trajectories)
# ---------------------------------------------------------
class ActionEncoder(nn.Module):
    def __init__(self, action_dim, chunk_size, latent_dim=32):
        super().__init__()
        # Compress the entire action trajectory [B, 8, 6] into a Gaussian distribution (mu, sigma)
        self.net = nn.Sequential(
            nn.Linear(action_dim * chunk_size, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim * 2) # Output mean and variance
        )

    def forward(self, action_trajectory):
        flat_action = action_trajectory.flatten(start_dim=1)
        h = self.net(flat_action)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

# ---------------------------------------------------------
# 2. ACT Backbone (Transformer)
# ---------------------------------------------------------
class ACTPolicy(nn.Module):
    def __init__(self, d_model=256, action_dim=6, chunk_size=8):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        
        # Visual/instruction projection
        self.img_proj = nn.Linear(1024, d_model)
        
        # Action decoder (Transformer)
        # ACT uses Transformer Encoder as a unique structure for Decoder
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=4)
        
        # Query vector (Learned Queries): corresponds to the 8 action steps we want to predict
        self.query_embed = nn.Embedding(chunk_size, d_model)
        
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, img_features, latent_z=None):
        batch_size = img_features.size(0)
        
        # 1. Prepare input token: [image token + latent random variable z]
        z_img = self.img_proj(img_features) # [B, 16, 256]
        
        # If it's inference phase, latent_z is all 0 or from prior distribution
        if latent_z is None:
            latent_z = torch.zeros(batch_size, 1, 256).to(img_features.device)
        
        # 2. Prepare Query: we want to predict the next 8 steps, so we need 8 query positions
        queries = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1) # [B, 8, 256]
        
        # 3. Concatenate all tokens and feed to Transformer
        # ACT's core: let Queries focus on image and latent variable
        input_tokens = torch.cat([queries, z_img, latent_z], dim=1) 
        output_tokens = self.transformer(input_tokens)
        
        # 4. Only take the first chunk_size tokens and map back to action space
        action_out = self.output_head(output_tokens[:, :self.chunk_size, :])
        return action_out

    def output_head(self, x):
        return self.action_head(x)


# ---------------------------------------------------------
# 3. Verification
# ---------------------------------------------------------
if __name__ == "__main__":
    batch_size = 2
    action_dim = 6
    chunk_size = 8

    # ----- ActionEncoder -----
    encoder = ActionEncoder(action_dim, chunk_size, latent_dim=32)
    fake_trajectory = torch.randn(batch_size, chunk_size, action_dim)
    mu, logvar = encoder(fake_trajectory)
    print(f"[ActionEncoder] mu shape: {mu.shape}, logvar shape: {logvar.shape}")

    # ----- ACTPolicy (inference with latent_z=None) -----
    policy = ACTPolicy(d_model=256, action_dim=action_dim, chunk_size=chunk_size)
    fake_img = torch.randn(batch_size, 16, 1024)

    action_out = policy(fake_img, latent_z=None)
    print(f"[ACTPolicy] Output shape: {action_out.shape}")
    print("ACT (CVAE Encoder + Transformer Decoder) verified!")