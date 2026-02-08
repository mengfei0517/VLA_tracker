# Data_Engine | 数据环

> 解决「垃圾进，垃圾出」的问题，建立标准化的数据流水线

---

## 🎯 核心目标

掌握机器人学习的数据工程范式，建立可复现、可扩展的数据采集与处理流水线。

---

## 📌 重点内容

### 1. LeRobot Ecosystem

**掌握 lerobot.datasets 的 Zarr 存储格式**

- **Zarr 格式优势**：分块存储、懒加载、多进程友好
- **数据结构规范**：
  - `observation.images.top` / `observation.images.wrist`
  - `action`：chunk 维度与时间对齐
  - `episode_index` / `frame_index`
- **数据处理接口**：`lerobot.datasets` 的 Dataset 与 DataLoader 用法
- **数据增强**：视觉 + 动作空间的同步变换

**这是目前工程落地最推崇的格式**，与 HuggingFace 生态深度集成。

### 2. UMI Implementation

**解析手持相机采集轨迹的数学坐标变换**

- **Camera-to-Gripper 变换**：
  - 外参标定（Hand-Eye Calibration）
  - 相机坐标系 → 机械臂基座坐标系
- **手持 vs 固定相机**：轨迹补偿与时间同步
- **多相机融合**：多视角下的位姿估计
- **UMI 遥操作**：异构设备的统一接口设计

---

## 📂 参考项目

| 项目 | 机构 | 学习重点 |
|------|------|----------|
| **LeRobot** | HuggingFace | 核心参考，尤其是其数据处理接口与 Zarr 规范 |
| **UMI** | Columbia | 异构遥操作的工程标杆，手持采集的数学建模 |

---

## 📖 建议学习顺序

1. 安装 LeRobot，跑通官方 demo 数据集
2. 理解 Zarr 存储结构的元数据布局
3. 编写自定义数据集 → LeRobot 格式的转换脚本
4. 研究 UMI 论文中的坐标变换推导
5. 实践：用手持手机采集一段轨迹并完成标定

---

## 🔗 延伸阅读

- LeRobot: [GitHub](https://github.com/huggingface/lerobot) | [Docs](https://huggingface.co/docs/lerobot)
- UMI: [GitHub](https://github.com/real-stanford/universal_manipulation_interface) | [Paper](https://arxiv.org/abs/2402.10329)
