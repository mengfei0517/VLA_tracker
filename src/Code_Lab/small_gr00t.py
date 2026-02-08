import torch
import torch.nn as nn

# ---------------------------------------------------------
# System 1: Low-Level Controller (small brain/instinct)
# Responsible for high-frequency motor control, usually trained in simulation environments using reinforcement learning
# ---------------------------------------------------------
class System1_LowLevel(nn.Module):
    def __init__(self, obs_dim=128, action_dim=20): # 20-dimensional may correspond to全身关节
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(), # RL commonly uses ELU
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim) # Output joint angles or torques
        )

    def forward(self, proprioception, task_embedding):
        """
        proprioception: the pose state of the robot (proprioception)
        task_embedding: the high-level instruction feature from System 2
        """
        x = torch.cat([proprioception, task_embedding], dim=-1)
        return self.net(x)

# ---------------------------------------------------------
# System 2: High-Level Planner (brain/cognition)
# This is a multi-modal large model (VLM) that is responsible for talking about pictures and decomposing logic
# ---------------------------------------------------------
class System2_HighLevel(nn.Module):
    def __init__(self, vlm_backbone):
        super().__init__()
        self.vlm = vlm_backbone # It could be part of CLIP, DinoV2, or Llama

    def forward(self, image, instruction):
        """
        Input image and speech, output a "task description vector"
        For example: extract "move to (x,y) and execute grab trajectory 05" from "help me get a Coke"
        """
        # Simplified example: output a feature to guide System 1
        task_latent = self.vlm(image, instruction)
        return task_latent

# ---------------------------------------------------------
# Mini GR00T Wrapper
# ---------------------------------------------------------
class MiniGR00T(nn.Module):
    def __init__(self, system1, system2):
        super().__init__()
        self.brain = system2  # Brain: slow reasoning (e.g., 5-10Hz)
        self.reflex = system1 # Reflex: fast execution (e.g., 100-500Hz)

    def step(self, observation):
        # 1. Brain analyzes the environment
        task_latent = self.brain(observation['image'], observation['goal'])
        # 2. Small brain, based on the brain's blueprint, combines body sensations, and outputs motor commands
        motor_action = self.reflex(observation['state'], task_latent)
        return motor_action


# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------
if __name__ == "__main__":
    batch_size = 2
    # System1 input: proprio (e.g. 64) + task_embed (64) -> 128 = obs_dim
    proprio_dim, task_embed_dim = 64, 64
    obs_dim = proprio_dim + task_embed_dim
    action_dim = 20

    # Mock VLM: image [B, 3, H, W], instruction [B, seq, dim] -> task_latent [B, task_embed_dim]
    class MockVLM(nn.Module):
        def __init__(self, out_dim=64):
            super().__init__()
            self.proj = nn.Linear(3 * 32 * 32 + 64, out_dim)

        def forward(self, image, instruction):
            # Flatten image; instruction: take mean or last token
            img_flat = image.flatten(1)
            inst_flat = instruction.mean(1) if instruction.dim() == 3 else instruction
            return self.proj(torch.cat([img_flat, inst_flat], dim=-1))

    system2 = System2_HighLevel(MockVLM(out_dim=task_embed_dim))
    system1 = System1_LowLevel(obs_dim=obs_dim, action_dim=action_dim)
    gr00t = MiniGR00T(system1, system2)

    # Fake observation
    obs = {
        "image": torch.randn(batch_size, 3, 32, 32),
        "goal": torch.randn(batch_size, 10, 64),
        "state": torch.randn(batch_size, proprio_dim),
    }

    action = gr00t.step(obs)
    print(f"[MiniGR00T] action shape: {action.shape}")
    print("GR00T (System1 + System2) verified!")