import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np

# ==========================================
# 1. 模拟配置文件 (通常从 norm_stats.json 加载)
# ==========================================
# 假设 A 是 14 DoF，B 是 16 DoF，统一向量长度为 34
CONFIG = {
    "robot_A": {
        "q02": np.array([-1.5] * 14 + [0.0] * 20), # 2% 分位数
        "q98": np.array([1.5] * 14 + [0.0] * 20),  # 98% 分位数
        "mask": [1] * 14 + [0] * 20,                # 只有前14位有效
        "sample_count": 1000                        # A数据很多
    },
    "robot_B": {
        "q02": np.array([-0.8] * 16 + [0.0] * 18),
        "q98": np.array([0.8] * 16 + [0.0] * 18),
        "mask": [1] * 16 + [0] * 18,                # 前16位有效
        "sample_count": 50                          # B数据很少（只有50条）
    }
}

# ==========================================
# 2. 跨本体数据集实现 (含语义对齐归一化)
# ==========================================
class CrossEmbodimentDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data = []
        # 模拟生成一些随机原始数据
        for r_id, info in config.items():
            for _ in range(info["sample_count"]):
                # 模拟原始动作数据 (Raw Actions)
                raw_action = np.random.uniform(-2, 2, size=34) 
                self.data.append((raw_action, r_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_action, r_id = self.data[idx]
        info = self.config[r_id]
        
        # 转换为 Tensor
        action = torch.tensor(raw_action, dtype=torch.float32)
        q02 = torch.tensor(info["q02"], dtype=torch.float32)
        q98 = torch.tensor(info["q98"], dtype=torch.float32)
        mask = torch.tensor(info["mask"], dtype=torch.float32)

        # --- 步骤 1: 语义对齐归一化 (2-98% -> [-1, 1]) ---
        # 公式: norm = 2 * (x - q02) / (q98 - q02) - 1
        eps = 1e-8
        norm_action = 2 * (action - q02) / (q98 - q02 + eps) - 1
        
        # --- 步骤 2: 离群值截断 (Clamp) ---
        norm_action = torch.clamp(norm_action, -1, 1)

        # --- 步骤 3: 掩码 (Padding 位清零) ---
        norm_action = norm_action * mask

        return {
            "action": norm_action,
            "mask": mask,
            "robot_id": r_id
        }

# ==========================================
# 3. 加权采样逻辑 (Balanced Sampler)
# ==========================================
def create_balanced_sampler(dataset):
    robot_ids = [d[1] for d in dataset.data]
    
    # 计算每个类别的频率
    counts = {}
    for rid in robot_ids:
        counts[rid] = counts.get(rid, 0) + 1
        
    # 计算权重: 权重与样本数成反比
    weights = [1.0 / counts[rid] for rid in robot_ids]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler

# ==========================================
# 4. 损失函数实现 (Masked MSE)
# ==========================================
def masked_mse_loss(pred, target, mask):
    # 计算原始 MSE
    loss = F.mse_loss(pred, target, reduction='none')
    # 只保留有效维度的 Loss
    masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)
    return masked_loss

# ==========================================
# 运行演示
# ==========================================
if __name__ == "__main__":
    # 实例化
    dataset = CrossEmbodimentDataset(CONFIG)
    sampler = create_balanced_sampler(dataset)
    loader = DataLoader(dataset, batch_size=8, sampler=sampler)

    # 模拟训练循环
    for batch in loader:
        actions = batch["action"]  # 归一化后的数据
        masks = batch["mask"]      # 对应的有效位
        r_ids = batch["robot_id"]

        print(f"Batch Robot IDs: {r_ids}")
        print(f"Normalized Action Sample (First 5 dims):\n{actions[0][:5]}")
        
        # 模拟模型输出
        mock_output = torch.randn_like(actions)
        
        # 计算带掩码的 Loss
        loss = masked_mse_loss(mock_output, actions, masks)
        print(f"Masked Loss: {loss.item():.4f}\n")
        
        break # 演示一个 Batch 即可