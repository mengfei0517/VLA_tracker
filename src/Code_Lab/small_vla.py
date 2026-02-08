import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# Cross-Attention Action Head: FM fuses visual tokens via attention
# ---------------------------------------------------------
class CrossAttentionActionHead(nn.Module):
    """
    Let action features (Query) attend to visual/text tokens (Key, Value).
    Each action step can selectively focus on different visual regions.
    """
    def __init__(self, action_dim, cond_dim, n_heads=4, hidden_dim=256):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.kv_proj = nn.Linear(cond_dim, hidden_dim)
        self.q_proj = nn.Linear(action_dim, hidden_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x_t, t, transformer_outputs):
        """
        x_t: [B, Chunk_size, Action_dim] - current trajectory state
        t: [B, 1] - timestep
        transformer_outputs: [B, Seq_len, Cond_dim] - visual/text tokens from Transformer
        """
        B, S, D = x_t.shape

        t_embed = self.time_mlp(t.view(-1, 1)).unsqueeze(1)  # [B, 1, hidden_dim]
        query = self.q_proj(x_t) + t_embed  # [B, S, hidden_dim]

        kv = self.kv_proj(transformer_outputs)  # [B, Seq_len, hidden_dim]

        attn_out, _ = self.cross_attn(query, kv, kv)  # [B, S, hidden_dim]

        v_pred = self.output_net(attn_out)  # [B, S, action_dim]
        return v_pred


# Reuse our previous FM network, but encapsulate it as the Action Head of VLA
class FlowActionHead(nn.Module):
    def __init__(self, action_dim, cond_dim, hidden_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(action_dim + cond_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x_t, t, cond):
        t_embed = self.time_mlp(t.view(-1, 1))
        # cond is the feature output from Transformer
        combined = torch.cat([x_t.flatten(start_dim=1), cond, t_embed], dim=-1)
        return self.net(combined).view_as(x_t)

# ---------------------------------------------------------
# Mini VLA complete model
# ---------------------------------------------------------
class MiniVLA(nn.Module):
    def __init__(self, d_model=256, n_heads=4, action_dim=6, chunk_size=8):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # 1. Modal projection layer (Projectors)
        # Simulate: the image after ViT is 1024-dimensional, and the instruction after BERT is 768-dimensional
        self.img_proj = nn.Linear(1024, d_model)
        self.txt_proj = nn.Linear(768, d_model)

        # 2. Transformer backbone (understanding layer)
        # Here we directly use PyTorch's native EncoderLayer to simulate the logic we wrote by hand earlier
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 3. Flow Matching head (execution layer)
        self.action_head = FlowActionHead(action_dim * chunk_size, d_model)

    def get_condition(self, img_embeds, txt_embeds):
        """
        Brain logic: fuse multi-modal information
        """
        # Map the image and text to a unified space
        z_img = self.img_proj(img_embeds) # [B, Seq_img, D]
        z_txt = self.txt_proj(txt_embeds) # [B, Seq_txt, D]
        
        # Concatenate token sequences
        tokens = torch.cat([z_img, z_txt], dim=1) # [B, Seq_total, D]
        
        # Through Transformer for self-attention interaction
        refined_tokens = self.transformer(tokens)
        
        # Take the first token (similar to CLS token) or global average as "global semantic instruction"
        cond = refined_tokens.mean(dim=1) 
        return cond

    def predict_action(self, cond, steps=3):
        """
        Cerebellum logic: Flow Matching inference
        """
        device = cond.device
        batch_size = cond.shape[0]
        
        # Starting point: random noise
        x_t = torch.randn((batch_size, self.chunk_size, self.action_dim)).to(device)
        dt = 1.0 / steps
        
        for i in range(steps):
            t = torch.full((batch_size, 1), i * dt).to(device)
            v_pred = self.action_head(x_t, t, cond)
            x_t = x_t + v_pred * dt # Euler integration: move forward in a straight line
            
        return x_t


# ---------------------------------------------------------
# Advanced VLA: FM + Cross-Attention to fuse visual tokens
# ---------------------------------------------------------
class AdvancedVLA(nn.Module):
    """
    Same backbone as MiniVLA, but uses Cross-Attention Action Head.
    FM step can selectively attend to different visual/text regions per action.
    """
    def __init__(self, d_model=256, n_heads=4, action_dim=6, chunk_size=8, hidden_dim=256):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        self.img_proj = nn.Linear(1024, d_model)
        self.txt_proj = nn.Linear(768, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Cross-Attention Head: action queries visual tokens
        self.action_head = CrossAttentionActionHead(
            action_dim=action_dim, cond_dim=d_model, n_heads=n_heads, hidden_dim=hidden_dim
        )

    def get_condition(self, img_embeds, txt_embeds):
        """Return full token sequence [B, Seq_len, D], not global mean."""
        z_img = self.img_proj(img_embeds)
        z_txt = self.txt_proj(txt_embeds)
        tokens = torch.cat([z_img, z_txt], dim=1)
        return self.transformer(tokens)

    def predict_action(self, transformer_outputs, steps=3):
        """
        FM inference: each action step attends to visual/text tokens via Cross-Attention.
        """
        device = transformer_outputs.device
        batch_size = transformer_outputs.shape[0]

        x_t = torch.randn((batch_size, self.chunk_size, self.action_dim)).to(device)
        dt = 1.0 / steps

        for i in range(steps):
            t = torch.full((batch_size, 1), i * dt).to(device)
            v_pred = self.action_head(x_t, t, transformer_outputs)
            x_t = x_t + v_pred * dt

        return x_t

# ---------------------------------------------------------
# Verify MVP
# ---------------------------------------------------------
if __name__ == "__main__":
    fake_img = torch.randn(1, 16, 1024)
    fake_txt = torch.randn(1, 5, 768)

    # ----- MiniVLA: global mean cond -----
    vla = MiniVLA()
    condition = vla.get_condition(fake_img, fake_txt)
    print(f"[MiniVLA] Cond shape (global mean): {condition.shape}")  # [1, 256]
    action = vla.predict_action(condition, steps=3)
    print(f"[MiniVLA] Action shape: {action.shape}")

    # ----- AdvancedVLA: Cross-Attention on full visual tokens -----
    adv = AdvancedVLA()
    tokens = adv.get_condition(fake_img, fake_txt)
    print(f"\n[AdvancedVLA] Token shape (full seq): {tokens.shape}")  # [1, 21, 256]
    action_adv = adv.predict_action(tokens, steps=3)
    print(f"[AdvancedVLA] Action shape: {action_adv.shape}")
    print("\nMini VLA + AdvancedVLA (FM + Cross-Attention) verified!")