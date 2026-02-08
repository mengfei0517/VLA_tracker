# VLA-FullStack-Guide 落地路线图

> Vision-Language-Action 全栈工程化指南：从底层原理到工程落地的完整路线图

---

## 📋 路线图总览

```
VLA-FullStack-Guide/
├── Conceptual_Framework/    # 1️⃣ 底层原理：建立 VLA 世界观
├── Data_Engine/             # 2️⃣ 数据环：标准化数据流水线
├── Training_&_Inference/    # 3️⃣ 算法黑盒：生成式策略数学关
├── Code_Lab/                # 4️⃣ 手速练习：拒绝调包侠
└── 工程化落地工具/          # 5️⃣ 工具链：转化、加速、评估
```

---

## 🎯 学习路径建议

| 阶段 | 模块 | 预计周期 | 前置依赖 |
|------|------|----------|----------|
| Phase 1 | Conceptual_Framework | 2-3 周 | 深度学习基础 |
| Phase 2 | Data_Engine | 2 周 | Phase 1 |
| Phase 3 | Training_&_Inference | 3-4 周 | Phase 1-2 |
| Phase 4 | Code_Lab | 持续 | Phase 1-3 |
| Phase 5 | 工程化落地工具 | 持续 | Phase 2+ |

---

## 🧠 核心思想

**"大脑"与"小脑"的辩证统一**  
- **大脑**：VLM 负责理解、规划、推理（What & Why）  
- **小脑**：Action Head 负责精密的运动控制（How）

VLA 的本质是让语言模型学会「说」动作，而不仅仅是「说」语言。

---

## 📚 参考资料索引

- [RT-2](https://github.com/google-deepmind/rt-2) | [OpenVLA](https://github.com/openvla/openvla)
- [LeRobot](https://github.com/huggingface/lerobot) | [UMI](https://github.com/real-stanford/universal_manipulation_interface)
- [Diffusion Policy](https://github.com/columbia-ai-robotics/diffusion_policy) | [TorchCFM](https://github.com/Atanov/torch-cfm)
- [ACT](https://github.com/tonyzhaozh/act) | [DORA-rs](https://github.com/ehsan4/dora-rs)
