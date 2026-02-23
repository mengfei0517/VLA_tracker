# 📝 Cross-Embodiment Robot Data Fusion & Training Technical Guide

## I. Core Challenges & Background

In multi-robot collaborative training (e.g., using the **pi0** architecture), we must address data consistency issues arising from different configurations (e.g., 14 DoF vs. 16 DoF) and heterogeneous observation spaces (mixed Cartesian and Joint states). The goal is to achieve a "**1+1 > 2**" effect, where cross-embodiment data enhances the generalization performance of a single-robot configuration.

---

## II. Action Space Architecture Design

### 1. Unified Action Vector

* **Container Design**: Define a fixed-length vector (e.g., 34 dimensions) to serve as a universal container for all robots.
* **Semantic Mapping**:
* **Non-Sequential Padding**: Avoid filling the vector strictly from index 0.
* **Fixed Slots**: Ensure joints with the same physical meaning occupy the same indices across all robots. For example:
* `Index 0-2`: End-effector translation increments ()
* `Index 3-5`: End-effector orientation increments ()
* `Index 33`: Gripper state




* **Loss Masking**:
* During training, set the Loss weight to 0 for dimensions that do not exist in a specific configuration.
* **Significance**: Prevents the model from "learning to output zero" for missing joints, thereby protecting the transfer of cross-embodiment physical priors.



---

## III. Normalization Strategy: Semantic Alignment & Independent Statistics

### 1. Semantic-aligned Normalization

* **Goal**: Convert raw physical values into "Action Intent."
* **Incremental Control**: Prioritize **incremental (Delta)** control for Cartesian space instead of absolute coordinates.
* **State Mapping**:
* Map "Fully Closed" gripper to .
* Map "Fully Open" gripper to .



### 2. Per-Robot Statistics (Independent Calculation)

* **Config-specific Stats**: Calculate the mean and standard deviation for Robot A and Robot B separately.
* **1-99% Percentile Clipping**:
* Use the 1st and 99th percentiles as normalization boundaries instead of simple Min-Max.
* **Formula**: 
* **Significance**: Effectively filters out outliers (e.g., sensor noise or collision spikes), ensuring the majority of valid data is distributed within .



---

## IV. Training Optimization Strategies

### 1. Balanced Sampling

* **Problem**: Imbalanced dataset scales (e.g., Dataset A is much larger than B).
* **Implementation**: Use `WeightedRandomSampler`.
* **Formula**: Sample Weight .
* **Effect**: Forces the model to see low-data configurations multiple times per epoch, preventing the model from being biased toward the dominant dataset.

### 2. Embodiment Conditioning

* **Strategy**: Incorporate an `Embodiment ID` at the input stage (e.g., One-hot encoding or Text Prompt).
* **Effect**: Provides the model with "self-awareness," allowing it to adjust its output logic based on the specific robot configuration it is currently controlling, even when visual inputs are similar.

---

## V. Engineering Implementation (PyTorch Style)

```python
import torch
import torch.nn.functional as F

class CrossEmbodimentDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, stats_config):
        self.data = data_list  # List of [(action, robot_id), ...]
        self.stats = stats_config # Contains q01, q99, and mask for each robot_id

    def __getitem__(self, idx):
        action, robot_id = self.data[idx]
        s = self.stats[robot_id]
        
        # 1. Semantic-aligned Normalization (1-99% Scaling)
        q01, q99 = torch.tensor(s['q01']), torch.tensor(s['q99'])
        norm_action = 2 * (action - q01) / (q99 - q01 + 1e-8) - 1
        
        # 2. Outlier Clipping and Masking
        norm_action = torch.clamp(norm_action, -1, 1)
        mask = torch.tensor(s['mask'])
        
        return {"action": norm_action * mask, "mask": mask, "robot_id": robot_id}

# Training Loss Calculation Example
def compute_loss(pred, target, mask):
    # Calculate MSE Loss only for valid joints/slots
    mse = F.mse_loss(pred, target, reduction='none')
    return (mse * mask).sum() / (mask.sum() + 1e-8)

```

---

## VI. Pitfalls & Best Practices

* **Clamping is Mandatory**: After 1-99% normalization, failing to `clamp(-1, 1)` will lead to loss spikes caused by extreme outliers.
* **Unit Alignment**: Before calculating statistics, ensure all configurations use unified units (Radians `Rad` is highly recommended for angles).
* **Coordinate System Verification**: Ensure Cartesian directions () represent consistent visual semantics across different robots (e.g.,  should always mean "Left" in the camera frame).
