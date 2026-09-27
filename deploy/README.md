# 树莓派部署指南

> 这个 `deploy/` 文件夹包含了可以直接拷贝到树莓派上运行的推理脚本和配套文件，**完全不依赖 tf2tutorial 包**，只需要 tflite-runtime 和 Pillow。

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `pi_classify.py` | 核心推理脚本，支持单图推理、基准测试、摄像头实时推理三种模式 |
| `labels_flowers.txt` | tf_flowers 数据集的 5 类花朵标签（daisy / dandelion / roses / sunflowers / tulips） |
| `README.md` | 本文档，部署和使用说明 |

---

## 怎么传到树莓派上

### 方法一：scp 命令（推荐）

在你的**电脑**上执行（不是树莓派上），把整个 deploy 文件夹传到树莓派：

```bash
# 上传整个 deploy 文件夹
scp -r deploy/ pi@raspberrypi.local:/home/pi/

# 只上传单个文件
scp deploy/pi_classify.py pi@raspberrypi.local:/home/pi/
scp deploy/labels_flowers.txt pi@raspberrypi.local:/home/pi/
```

如果不知道树莓派的 IP，可以登录路由器后台查看，或者用 hostname 尝试：
- 默认主机名：`raspberrypi.local`
- 默认用户名：`pi`

### 方法二：U盘

1. 把文件拷到 U 盘
2. 插到树莓派上
3. 挂载并复制：

```bash
lsblk                           # 查看设备，通常是 /dev/sda1
sudo mkdir /mnt/usb
sudo mount /dev/sda1 /mnt/usb
cp /mnt/usb/deploy/* ~/
sudo umount /mnt/usb
```

### 方法三：VS Code Remote-SSH（开发体验最好）

1. VS Code 安装 **Remote - SSH** 扩展
2. F1 → `Remote-SSH: Connect to Host`
3. 输入 `pi@raspberrypi.local`
4. 直接在 VS Code 里编辑、运行、传输文件

---

## 环境准备

### 1. 安装依赖

在树莓派上执行：

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装 pip（如果没有的话）
sudo apt install python3-pip -y

# 安装 TFLite 运行时（轻量，推荐）
pip install tflite-runtime

# 安装 Pillow（图像处理）
pip install Pillow

# 摄像头模式还需要 OpenCV（可选）
pip install opencv-python-headless
```

### 2. 准备模型文件

你需要一个 `.tflite` 模型文件。可以从以下途径获取：

- **本项目第 10 章生成的模型**：在 PC 上跑完第 10 章后，模型保存在 `models/` 目录下，例如 `chapter10_mobilenet_int8.tflite`
- **官方预训练模型**：从 TensorFlow Hub 或 TFLite 模型库下载
- **你自己训练的模型**：用 TensorFlow Lite Converter 转换得到

把 `.tflite` 文件和对应的标签文件传到树莓派上即可。

---

## 三种使用方式

### 方式一：单张图片推理

最基本的用法，对一张图片做分类推理：

```bash
python3 pi_classify.py \
  --model model.tflite \
  --image test.jpg \
  --labels labels_flowers.txt \
  --top_k 3
```

输出示例：

```
============================================================
  TFLite 图像分类推理工具 (pi_classify)
============================================================
  运行时 : tflite_runtime
  架构   : ARM64 (aarch64) — 64 位 ARM 架构
  系统   : Linux 6.1.0
  提示   : 检测到 ARM64 架构，推荐使用 int8 量化模型 + 4 线程

  标签文件: labels_flowers.txt (5 类)
  图片文件: sunflower.jpg
  图片尺寸: 800x600

  模型文件: model.tflite (2.35 MB)
  线程数  : 4

  —— 模型输入信息 ——
    [0] name=serving_default_input_1:0
        shape=[1, 160, 160, 3], dtype=float32

  —— 模型输出信息 ——
    [0] name=StatefulPartitionedCall:0
        shape=[1, 5], dtype=float32

  推理中...

  推理耗时: 85.32 ms

  —— Top-3 结果 ——

  #1  sunflowers                 92.45%  [█████████████████████████████░░░]
  #2  daisy                       4.12%  [█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  #3  dandelion                   2.30%  [█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]

============================================================
  完成
============================================================
```

### 方式二：基准测试

跑 N 次推理，输出平均延迟和 FPS，用来测试性能：

```bash
python3 pi_classify.py \
  --model model.tflite \
  --image test.jpg \
  --benchmark 100 \
  --num_threads 4
```

输出示例：

```
  —— 基准测试（100 次）——

  预热中... ✓
  进度: 20/100
  进度: 40/100
  进度: 60/100
  进度: 80/100
  进度: 100/100

  ┌──────────────────────────────────────┐
  │         基准测试结果                  │
  ├──────────────────┬───────────────────┤
  │ 指标             │            数值   │
  ├──────────────────┼───────────────────┤
  │ 平均延迟         │        45.23 ms   │
  │ 标准差           │         2.15 ms   │
  │ 最小延迟         │        42.08 ms   │
  │ 最大延迟         │        52.67 ms   │
  │ 吞吐量 (FPS)     │             22.11 │
  └──────────────────┴───────────────────┘
```

### 方式三：摄像头实时推理

用摄像头实时采集画面并推理，需要安装 OpenCV：

```bash
python3 pi_classify.py \
  --model model.tflite \
  --camera 0 \
  --labels labels_flowers.txt \
  --num_threads 4
```

按 `Ctrl+C` 退出。

输出示例（每秒更新一行）：

```
  FPS:  12.3  |  Top-1: sunflowers (89.5%)
```

---

## 怎么用你自己的模型

非常简单，只需要两个文件：

1. **模型文件**：`your_model.tflite`
2. **标签文件**：`your_labels.txt`（每行一个类别名，顺序和模型输出一致）

然后运行：

```bash
python3 pi_classify.py \
  --model your_model.tflite \
  --image test.jpg \
  --labels your_labels.txt
```

### 标签文件格式

纯文本文件，每行一个类别名，顺序必须和模型输出的类别索引一致：

```
daisy
dandelion
roses
sunflowers
tulips
```

即：第 0 行对应输出索引 0，第 1 行对应输出索引 1，以此类推。

### 模型要求

- 输入：图像分类模型，输入 shape 为 `[1, H, W, 3]`
- 输出：分类概率，输出 shape 为 `[1, num_classes]`
- 支持的量化格式：float32、float16、int8

如果你的模型输入不是 `[0, 1]` 范围（比如 MobileNetV2 预训练用的是 `[-1, 1]`），可能需要在 `preprocess_image` 函数中调整归一化方式。

---

## 性能参考

以下数据为参考值（MobileNetV2 160x160，4 线程），实际性能受模型结构、输入尺寸、系统负载、温度等因素影响。

| 设备 | float32 | int8 量化 | 说明 |
|------|---------|----------|------|
| **树莓派 5** (2.4GHz Cortex-A76) | ~30 ms / 33 FPS | ~12 ms / 83 FPS | 性能最好 |
| **树莓派 4B** (1.5GHz Cortex-A72) | ~80 ms / 12 FPS | ~35 ms / 29 FPS | 性价比之选 |
| **树莓派 Zero 2 W** (1GHz Cortex-A53) | ~300 ms / 3 FPS | ~120 ms / 8 FPS | 适合低功耗场景 |
| **树莓派 Zero W** (单核 1GHz) | ~2000 ms / 0.5 FPS | ~800 ms / 1.2 FPS | 不推荐，太慢 |

### 性能优化建议

1. **使用 int8 量化模型**：速度提升 2-3 倍，体积缩小 4 倍
2. **调整输入分辨率**：224 → 160 → 128 → 96，速度和像素数成正比
3. **多线程推理**：`--num_threads 4`（4 核 CPU 上设为 4 最佳）
4. **使用更小的模型**：MobileNetV3-Small 比 MobileNetV2 快约 2 倍
5. **加散热**：温度过高会降频，影响性能
6. **终极方案**：Google Coral USB TPU，int8 模型可跑到 100+ FPS

---

## 故障排查

### 1. `ModuleNotFoundError: No module named 'tflite_runtime'`

**原因**：没有安装 tflite-runtime，或者装到了别的 Python 版本里。

**解决**：
```bash
python3 -m pip install tflite-runtime
```

如果还是找不到，确认架构：
```bash
uname -m   # 应该输出 aarch64（64位）或 armv7l（32位）
```
如果是 32 位系统（armv7l），建议重装 64 位系统。

---

### 2. 推理结果全错 / 置信度很低

**最可能的原因：图片预处理和训练时不一致。**

检查清单：
- 像素值范围：是 `[0, 1]`、`[-1, 1]` 还是 `[0, 255]`？
- 颜色通道顺序：RGB 还是 BGR？
- 图像尺寸：是不是模型要求的尺寸？

**快速排查**：在脚本里加一行打印输入数据范围：
```python
print(f"输入范围: {input_data.min()} ~ {input_data.max()}")
```

如果你的模型训练时用的是 `[-1, 1]` 归一化（如 MobileNetV2 预训练），需要修改 `preprocess_image` 函数中的归一化逻辑。

---

### 3. 摄像头打不开

**排查步骤**：
```bash
# 1. 检查设备文件
ls /dev/video*

# 2. 确认用户在 video 组
groups
# 如果没有 video，执行：sudo usermod -aG video $USER 然后重启

# 3. 测试摄像头（官方摄像头模块）
libcamera-hello

# 4. 如果是 USB 摄像头，换个 USB 口试试
```

---

### 4. 推理速度特别慢

**可能原因及解决方法**：

1. 用的是 float32 模型 → 换成 int8 量化模型
2. 没开多线程 → 加上 `--num_threads 4`
3. 输入分辨率太高 → 试试 160x160 或 96x96
4. 温度过高降频 → 加散热片或风扇
5. 用了完整 tensorflow 而不是 tflite-runtime → 安装 tflite-runtime

检查温度：
```bash
vcgencmd measure_temp
# 超过 80°C 会降频
```

---

### 5. 内存不足 / OOM

**现象**：程序被系统杀掉，或报 `MemoryError`。

**解决方法**：
1. 用 int8 量化模型（内存占用只有 float32 的 1/4）
2. 减小输入分辨率
3. 关闭其他占用内存的程序
4. 如果是 Pi Zero（512MB），考虑换更大内存的型号

---

## 相关资源

- [TensorFlow Lite 官方文档](https://www.tensorflow.org/lite)
- [树莓派官方文档](https://www.raspberrypi.com/documentation/)
- [本项目第 10 章：边缘计算与树莓派部署](../chapters/10_edge_raspberry_pi.py)
- [树莓派部署完整指南](../docs/raspberry_pi_deployment_zh.md)
