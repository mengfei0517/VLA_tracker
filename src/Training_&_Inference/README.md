# Training_&_Inference | 算法黑盒

> 攻克生成式策略（Generative Strategy）的数学关

---

## 🎯 核心目标

深度理解 Diffusion 与 Flow Matching 在机器人策略学习中的应用，掌握从噪声到动作的数学原理。

---

## 📌 重点内容

### 1. Diffusion Policy

**深度解析 Noise Scheduler 和 V-prediction**

- **DDPM 回顾**：前向扩散与反向去噪的数学形式
- **Noise Scheduler**：线性 vs 余弦，对训练稳定性的影响
- **V-prediction**：为何能解决轨迹发散问题？
  - 对比 ε-prediction 与 x-prediction
  - 方差膨胀与梯度尺度
- **Policy Head 设计**：chunk-based 输出、时间维度处理

### 2. Flow Matching

**学习直线向量场（Optimal Transport Path）**

- **从 Score Matching 到 Flow Matching**：ODE 视角
- **Optimal Transport**：π₀ 为何能在更少步数内生成更稳的动作？
- **Conditional Flow Matching**：观测条件如何注入
- **FM + Transformer 融合**：π₀ Blog 的最新思路

**核心优势**：相比 DDPM，Flow Matching 的采样路径更短、更直，推理效率更高。

---

## 📂 参考项目

| 项目 | 学习重点 |
|------|----------|
| **Diffusion Policy** | Policy Head 设计、Noise Scheduler 实现、V-prediction |
| **TorchCFM** | 条件流匹配的最简实现，理解 ODE 求解 |
| **π₀ (Pi-0) Blog** | FM + Transformer 融合思路，工程化实践 |

---

## 📖 建议学习顺序

1. 复现 Diffusion Policy 的 Policy Head，单步推理验证
2. 对比 ε / x / v 三种 prediction 的代码差异
3. 用 TorchCFM 跑通 Flow Matching 的 toy 任务
4. 阅读 π₀ Blog，理解 FM 在 VLA 中的应用
5. 实验：相同数据下，DP vs FM 的采样步数与成功率

---

## 🔗 延伸阅读

- Diffusion Policy: [GitHub](https://github.com/columbia-ai-robotics/diffusion_policy) | [Paper](https://arxiv.org/abs/2303.04137)
- TorchCFM: [GitHub](https://github.com/Atanov/torch-cfm)
- π₀ Blog: [Pi-0 Announcement](https://www.pi.ai/blog)
