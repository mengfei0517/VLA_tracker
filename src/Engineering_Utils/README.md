# 工程化落地工具

> 数据转化、推理加速、评估验证——从实验到生产的工具链

---

## 🎯 核心目标

建立可复用的工程化工具链，打通从原始数据到部署上线的完整闭环。

---

## 📌 重点内容

### 1. 数据转化工具

| 工具类型 | 功能描述 | 典型场景 |
|----------|----------|----------|
| **格式转换** | 原始采集数据 → LeRobot Zarr / RT-X 格式 | 多源数据统一 |
| **坐标系变换** | Camera / Gripper / World 标定与转换 | 手持 UMI 数据 |
| **轨迹清洗** | 异常检测、平滑、重采样 | 提升数据质量 |
| **数据切片** | 按 episode / task 切分与合并 | 增量数据管理 |

### 2. 推理加速工具

| 工具类型 | 功能描述 | 典型技术 |
|----------|----------|----------|
| **模型编译** | TorchScript / ONNX / TensorRT 导出 | 部署优化 |
| **量化** | FP16 / INT8 / 动态量化 | 边缘设备 |
| **剪枝** | 结构化剪枝、知识蒸馏 | 轻量化 |
| **批处理** | 多任务批推理、异步队列 | 吞吐优化 |

### 3. 评估验证工具

| 工具类型 | 功能描述 | 典型指标 |
|----------|----------|----------|
| **仿真评估** | Isaac Sim / MuJoCo 环境对接 | Success Rate |
| **真机评估** | 远程触发、结果回传、日志记录 | 实机成功率 |
| **离线评估** | 回放轨迹、动作分布分析 | 策略一致性 |
| **Benchmark** | 标准任务集、排行榜 | 横向对比 |

---

## 🛠️ 推荐工具栈

```
数据转化:  Python + Zarr + OpenCV + NumPy
推理加速:  ONNX Runtime / TensorRT / OpenVINO
评估验证:  Weights & Biases / MLflow / 自研 Dashboard
```

---

## 📖 建议实践顺序

1. 编写「任意格式 → LeRobot Zarr」的通用转换脚本
2. 将训练好的模型导出为 ONNX，对比推理时延
3. 搭建最小可用的真机评估流程（单任务 + 10 次 rollout）
4. 建立评估指标的自动化汇总与可视化

---

## 🔗 延伸阅读

- ONNX: [onnx.ai](https://onnx.ai/)
- TensorRT: [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- LeRobot 数据格式: [HuggingFace Datasets](https://huggingface.co/docs/lerobot)
