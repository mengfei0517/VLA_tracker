# # Data_Engine | High-Fidelity Robot Learning

> **Core Philosophy**: In VLA systems, the Action Head is only as precise as the Perceptual Grounding provided by the Data Engine. We shift from "Big Data" to **"Balanced, High-Density Data."**

## 🎯 Case Study: The "Peg-in-Hole" Precision Crisis

During our 30h  post-training, we identified a critical bottleneck: the model fails at the final assembly stage due to **Visual Ambiguity** and **Data Contradiction**.

### 1. The Redundancy Paradox (30h @ 2eps vs. 15h @ 4eps)

* **The Problem**: Why did 30h @ 2eps outperform 15h @ 4eps, yet 15h @ 4eps was better than 15h @ 2eps?
* **The Logic**:
* **Statistical Smoothing**: 30h of data provides a "thick" distribution. Even if redundant, the micro-variations in human demos act as a **Natural Low-Pass Filter**, creating a smoother Velocity Field in Flow Matching.
* **Information Density**: 15h of pruned data has higher "Information Per Batch." It requires more epochs (4) to extract the diverse features, whereas 30h is so dense that 4 epochs lead to **Overfitting on Noise**.
* **The Conclusion**: Scale provides a "Safety Net" of generalizability, but Pruning requires deeper training to reach the same "Stability."



### 2. The Perceptual Grounding Gap (Hand-Eye vs. Overhead)

* **The Problem**: Why does the model rely on the far-away overhead camera for precision tasks?
* **The Logic**: The overhead camera has a stable **Global Coordinate Frame**. Hand-eye cameras suffer from **Feature Drift** during movement. The Transformer naturally attends to the most stable signal, even if it lacks the resolution for  assembly.
* **The Solution**: **Multi-Scale Tokenization.** Force the model to "Zoom In."

---

## 🏗️ Technical Architecture: The Balanced Data Pipeline

To solve **State Aliasing** (model cannot distinguish between "Success" and "Failure" in blurry views), we implement a multi-source balanced pipeline.

### 1. Balanced Multi-Source Loader

This prevents the "Drowning Effect" where 30h of old data washes out 3h of new, high-precision corrective data.

```python
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset

def get_vla_balanced_loader(dataset_old, dataset_new, batch_size=32):
    """
    Implements 1:1 Balanced Sampling to prioritize high-precision corrections.
    """
    # Combine the redundant base and the sparse precision datasets
    combined_dataset = ConcatDataset([dataset_old, dataset_new])
    
    # Assign weights so that each dataset contributes 50% of every batch
    weights_old = [1.0 / len(dataset_old)] * len(dataset_old)
    weights_new = [1.0 / len(dataset_new)] * len(dataset_new)
    weights = torch.DoubleTensor(weights_old + weights_new)
    
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return DataLoader(combined_dataset, batch_size=batch_size, sampler=sampler)

```

### 2. Multi-View Token Fusion (RoI Injection)

Instead of just feeding raw frames, we inject a **Region of Interest (RoI)** token from the wrist camera to fix the resolution mismatch.

```python
def extract_precision_tokens(wrist_img, overhead_img, model_backbone):
    # 1. Global context from overhead
    global_feat = model_backbone(overhead_img) # [B, N_global, D]
    
    # 2. Precision crop from wrist camera (The "Zoom-In" effect)
    # This forces the model to see the hole edges clearly
    wrist_crop = transforms.CenterCrop(112)(wrist_img) 
    local_feat = model_backbone(wrist_crop) # [B, N_local, D]
    
    # 3. Concatenate for Transformer input
    return torch.cat([global_feat, local_feat], dim=1)

```

---

## 📂 Engineering Roadmap & Best Practices

| Strategy | Goal | Why it works |
| --- | --- | --- |
| **Balanced Sampling** | Prevent Catastrophic Forgetting | Ensures the high-precision "Corrections" are seen as frequently as "Global Moves." |
| **Domain Prompting** | Resolve Strategy Conflict | Adding a `<precision_mode>` token tells the model when to ignore blurry global cues. |
| **Delta-Pos Injection** | Solve State Aliasing | Visuals fail at , but Relative Z-depth from proprioception is an absolute success indicator. |
| **Pruning + High Epochs** | Maximize Sample Efficiency | Pruning removes the "easy" redundant samples; higher epochs are then needed to master the "hard" ones. |

---

## 🔗 Critical Tooling & Links

* **[Hugging Face LeRobot](https://github.com/huggingface/lerobot)**: Standardizing the `LeRobotDataset` for multi-cam VLA training.
* **[ (Physical Intelligence)](https://www.physicalintelligence.company/blog/pi0)**: Reference for the Flow Matching objective and data scaling laws.
* **[UMI (Universal Manipulation Interface)](https://github.com/real-stanford/universal_manipulation_interface)**: Best practices for hand-eye camera calibration and data collection.
* **[OpenVLA Action Tokenization](https://github.com/openvla/openvla)**: Contrastive methods to improve action grounding.
