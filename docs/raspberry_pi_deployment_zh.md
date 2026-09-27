# 树莓派边缘计算实战指南：从零跑通第一个 TFLite 推理

> 面向计算机专业学生的实操手册，从"拿到一块树莓派"到"跑通第一个 TensorFlow Lite 推理"的完整步骤。
> 本文档配套 TensorFlow 教程项目使用，帮你把训练好的模型搬到真实硬件上。

---

## 一、准备工作

### 1.1 硬件选择

#### 推荐型号

| 型号 | 推荐指数 | CPU | 内存 | 价格参考 | 适合场景 |
|------|---------|-----|------|---------|---------|
| **树莓派 5 (8GB)** | ⭐⭐⭐⭐⭐ | 4核 Cortex-A76 2.4GHz | 8GB LPDDR4X | ~600元 | 性能最好，推理流畅，推荐预算充足的同学 |
| **树莓派 5 (4GB)** | ⭐⭐⭐⭐ | 4核 Cortex-A76 2.4GHz | 4GB LPDDR4X | ~500元 | 性价比之选，跑 TFLite 完全够用 |
| **树莓派 4B (4GB/8GB)** | ⭐⭐⭐⭐ | 4核 Cortex-A72 1.5GHz | 4GB/8GB LPDDR4 | ~350/450元 | 经典款，性能足够，二手市场多 |
| 树莓派 Zero 2 W | ⭐⭐ | 4核 Cortex-A53 1GHz | 512MB | ~150元 | 内存太小，跑模型会很吃力 |
| 树莓派 Zero W | ⭐ | 单核 1GHz | 512MB | ~100元 | **不推荐**，单核推理极慢 |

**为什么不推荐 Zero 系列？**

- 内存只有 512MB，系统本身就占掉一大半，留给模型的空间很少
- CPU 性能弱（A53/A7 架构），跑一次 224x224 的 MobileNet 可能需要 2-3 秒
- 没有足够的算力做实时推理，适合超简单的场景（比如纯传感器数据）

#### 为什么推荐 64 位？

- 64 位系统能访问全部 4GB/8GB 内存（32 位只能识别 ~3GB）
- TFLite 的 ARM64 优化更好，NEON 指令集利用更充分
- 大多数预编译的 tflite-runtime wheel 包优先支持 arm64
- 很多新的 AI 库（如 ONNX Runtime）已经放弃 32 位 ARM 支持

### 1.2 必备配件清单

| 配件 | 用途 | 选购建议 |
|------|------|---------|
| **Micro SD 卡** | 系统盘 | 至少 32GB，Class 10，U3 等级。推荐三星 EVO Plus / 闪迪至尊高速 |
| **电源适配器** | 供电 | Pi 5 用 27W USB-C PD 电源；Pi 4B 用 15W (5V 3A) USB-C 电源。**一定要买官方或品牌电源**，劣质电源会导致各种玄学问题 |
| **散热片/散热风扇** | 降温 | Pi 5 发热较大，推荐带风扇的主动散热。Pi 4B 被动散热片即可。推理时 CPU 满载，温度会飙升到 70°C+ |
| **摄像头模块** | 图像采集 | 官方 Camera Module 3（12MP）或第三方 USB 摄像头。新手推荐 USB 摄像头，插上去就能用 |
| **读卡器** | 刷系统用 | USB 3.0 读卡器，速度快很多 |

📝 **小贴士**：如果不确定买哪个，直接搜"树莓派 5 8GB 入门套件"，一般店家都会配好电源+散热+SD卡，省得自己一个个挑。

⚠️ **注意**：不要用手机充电器给树莓派供电！很多手机充电器是 5V 2A 的，电流不够。树莓派满载时会出现"闪电"图标（欠压警告），然后自动降频，推理速度会慢很多。

### 1.3 系统选择

**推荐：Raspberry Pi OS (Bookworm) 64-bit**

- 基于 Debian 12，软件源新
- 64 位，性能更好
- 官方维护，兼容性最好
- 桌面版（with desktop）适合新手，Lite 版适合无头运行

为什么不选 Ubuntu / Arch / 其他系统？

- 官方系统对硬件的支持最好（摄像头、GPIO、蓝牙等）
- 社区资料多，遇到问题容易搜到解决方案
- 省得折腾驱动，专注于算法本身

---

## 二、系统设置

### 2.1 刷系统

**工具：Raspberry Pi Imager（官方工具，免费）**

下载地址：https://www.raspberrypi.com/software/

**步骤：**

1. 打开 Raspberry Pi Imager
2. 点击"选择设备" → 选择你的树莓派型号
3. 点击"选择操作系统" → `Raspberry Pi OS (other)` → `Raspberry Pi OS (64-bit)`
   - 带桌面的选 **Raspberry Pi OS Desktop (64-bit)**
   - 想省空间、纯命令行的选 **Raspberry Pi OS Lite (64-bit)**
4. 点击"选择存储" → 选择你的 SD 卡
5. **重要：点击右下角的齿轮图标（高级设置）**，提前配置好：
   - 设置 hostname（比如 `raspberrypi.local`）
   - 启用 SSH（选"使用密码认证"）
   - 设置用户名和密码（**别用默认的 pi/raspberry**，不安全）
   - 配置 WiFi（输入你家的 WiFi 名称和密码）
   - 设置时区（Asia/Shanghai）
6. 点击"烧录"，等待完成

📝 **小贴士**：在高级设置里提前配好 WiFi 和 SSH，刷完系统插电就能直接 SSH 连上去，不用接显示器键盘，非常方便。

⚠️ **注意**：刷系统会格式化 SD 卡，里面的所有数据都会被清空，记得先备份！

### 2.2 首次开机与连接

1. 把 SD 卡插进树莓派
2. 插上网线（或等 WiFi 自动连接）
3. 插上电源，等待 1-2 分钟开机
4. 在你的电脑上打开终端 / PowerShell，用 SSH 连接：

```bash
# 格式：ssh 用户名@hostname.local
ssh pi@raspberrypi.local
```

如果连不上，可以登录路由器后台查看树莓派的 IP 地址，然后用 IP 连接：

```bash
ssh pi@192.168.1.100
```

📝 **小贴士**：Windows 用户如果 `ssh` 命令不可用，可以安装 PuTTY 或启用 Windows 的 OpenSSH 功能（设置 → 应用 → 可选功能 → 添加功能 → OpenSSH 客户端）。

### 2.3 启用摄像头、VNC 等功能

```bash
sudo raspi-config
```

进入配置界面后：

- **启用摄像头**：`Interface Options` → `Camera` → `Yes`
- **启用 VNC**（远程桌面）：`Interface Options` → `VNC` → `Yes`
- **启用 I2C/SPI**（如果接传感器）：`Interface Options` → `I2C` / `SPI` → `Yes`

设置完后选择 `Finish`，会提示重启，选 `Yes`。

### 2.4 扩容文件系统

刚刷完系统，SD 卡的空间可能没有全部用上。检查一下：

```bash
df -h
```

如果 `/` 分区只有几 GB，说明需要扩容：

```bash
sudo raspi-config
```

选择 `Advanced Options` → `Expand Filesystem`，然后重启。

重启后再用 `df -h` 确认，根分区应该已经占满整张 SD 卡了。

### 2.5 换国内源

官方源在国内访问很慢，换成清华源或中科大源，apt 和 pip 速度会快很多。

#### 更换 apt 源（清华源）

```bash
# 先备份原来的源列表
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak

# 编辑源列表
sudo nano /etc/apt/sources.list
```

把里面的内容全部替换为（Bookworm 版本）：

```
deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free non-free-firmware
deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free non-free-firmware
deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-backports main contrib non-free non-free-firmware
deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free non-free-firmware
```

按 `Ctrl+O` 保存，`Ctrl+X` 退出。

还有一个系统专用的源文件也要换：

```bash
sudo nano /etc/apt/sources.list.d/raspi.list
```

替换为：

```
deb https://mirrors.tuna.tsinghua.edu.cn/raspberrypi/ bookworm main
```

#### 更换 pip 源（清华源）

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 验证

```bash
# 更新 apt 缓存，测试速度
sudo apt update
```

如果速度明显变快（几 MB/s），说明换源成功。

### 2.6 更新系统

```bash
sudo apt update && sudo apt upgrade -y
```

这一步可能需要 10-20 分钟，取决于网络和 SD 卡速度。耐心等它跑完。

更新完后重启一下：

```bash
sudo reboot
```

---

## 三、安装 TFLite 运行时

### 3.1 方法一：pip 安装 tflite-runtime（推荐）

这是最简单、最轻量的方式，只安装推理需要的运行时，体积只有 ~10MB。

```bash
# 先确认 Python 版本和架构
python3 --version
uname -m
# 应该输出 aarch64（64位）或 armv7l（32位）
```

⚠️ **注意**：如果 `uname -m` 输出的是 `armv7l`，说明你装的是 32 位系统！建议重装 64 位系统，否则可能找不到预编译包。

安装：

```bash
pip install tflite-runtime
```

如果提示找不到匹配的版本，可以试试指定版本：

```bash
# 试试安装具体版本
pip install tflite-runtime==2.14.0
```

### 3.2 方法二：pip 安装 tensorflow（不推荐）

```bash
pip install tensorflow
```

**为什么不推荐？**

- 安装包很大（几百 MB），下载慢，安装也慢
- 占空间大（安装后 ~1GB）
- 树莓派上用不到训练功能，只需要推理
- 安装时间可能长达 30 分钟以上

只有当你需要用完整 TensorFlow 的某些功能（如 `tf.data` API）时，才考虑这种方式。

### 3.3 方法三：从源码编译（进阶，不推荐初学者）

如果你需要最新的功能或特定优化，可以从源码编译 TFLite。但编译时间很长（Pi 5 上可能需要 1-2 小时），而且容易出问题。

参考官方文档：https://www.tensorflow.org/lite/guide/build_arm64

初学者直接跳过就行，等以后有需求再研究。

### 3.4 验证安装

```bash
python3 -c "import tflite_runtime.interpreter as tflite; print('TFLite 运行时安装成功！')"
```

如果输出 `TFLite 运行时安装成功！`，恭喜你，安装完成！

### 3.5 常见坑

#### 坑1：架构不匹配（armhf vs arm64）

**症状**：`pip install tflite-runtime` 报错 `No matching distribution found`

**原因**：系统是 32 位的（armhf/armv7l），但 PyPI 上的 tflite-runtime 可能只有 64 位版本

**解决方法**：
- 首选：重装 64 位系统（一劳永逸）
- 备选：找第三方编译的 32 位 wheel 包，或用 apt 安装：`sudo apt install python3-tflite-runtime`

#### 坑2：Python 版本不对

**症状**：导入时报错，或找不到模块

**原因**：系统里有多个 Python 版本（2.7 / 3.9 / 3.11），pip 装到了别的版本里

**解决方法**：
```bash
# 确认当前 python3 版本
python3 --version

# 用 python3 -m pip 确保装到正确的版本
python3 -m pip install tflite-runtime

# 验证
python3 -c "import tflite_runtime.interpreter as tflite; print('ok')"
```

#### 坑3：numpy 版本冲突

**症状**：导入时出现 numpy 相关错误

**解决方法**：
```bash
pip install --upgrade numpy
```

---

## 四、传输模型和测试图片

把你在电脑上训练好的 `.tflite` 模型文件和测试图片传到树莓派上。

### 4.1 方法一：scp 命令（最快，推荐命令行玩家）

在你的**电脑**上执行（不是树莓派上）：

```bash
# 上传单个文件
scp model.tflite pi@raspberrypi.local:/home/pi/models/

# 上传整个文件夹
scp -r my_project/ pi@raspberrypi.local:/home/pi/

# 下载文件（从树莓派传到电脑）
scp pi@raspberrypi.local:/home/pi/result.jpg ./
```

### 4.2 方法二：U盘（最简单，适合大文件）

1. 把文件拷到 U 盘里
2. U 盘插到树莓派上
3. 挂载并复制：

```bash
# 查看 U 盘设备名
lsblk
# 通常是 /dev/sda1

# 创建挂载点
sudo mkdir /mnt/usb

# 挂载（FAT32格式）
sudo mount /dev/sda1 /mnt/usb

# 复制文件
cp /mnt/usb/model.tflite ~/models/

# 用完卸载（一定要卸载再拔！）
sudo umount /mnt/usb
```

### 4.3 方法三：VS Code Remote-SSH（推荐，开发体验最好）

如果你用 VS Code，强烈推荐装 Remote-SSH 插件：

1. 在 VS Code 里安装 **Remote - SSH** 扩展
2. 按 `F1`，输入 `Remote-SSH: Connect to Host`
3. 输入 `pi@raspberrypi.local`
4. 输密码，连接成功
5. 打开树莓派上的文件夹，就像编辑本地文件一样
6. 终端也直接是树莓派的终端

优点：
- 文件编辑、代码运行、文件传输一体化
- 可以直接拖拽文件
- 代码补全、调试都能用

📝 **小贴士**：第一次连接可能会慢一点，因为要在远端安装 VS Code Server。后面就快了。

### 4.4 目录结构建议

```
/home/pi/
├── models/              # 存放 .tflite 模型文件
│   ├── mobilenet_v2.tflite
│   └── my_custom_model.tflite
├── test_images/         # 测试图片
│   ├── cat.jpg
│   └── dog.jpg
├── projects/            # 你的项目代码
│   └── first_inference/
│       ├── inference.py
│       └── labels.txt
└── data/                # 采集的数据
```

---

## 五、第一个推理程序

我们来写一个完整的图像分类推理程序，大约 30 行代码。

### 5.1 准备模型和标签

如果你还没有自己的模型，可以先用官方的 MobileNet V2 模型来测试：

```bash
# 下载 MobileNet V2 量化模型和标签
cd ~/models
wget https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v2_1.0_224_quant.tflite
wget https://storage.googleapis.com/download.tensorflow.org/models/image_labels.txt
```

准备一张测试图片（随便找一张 jpg 图片即可），放到 `~/test_images/` 目录下。

### 5.2 完整代码

在 `~/projects/first_inference/` 下创建 `inference.py`：

```python
import numpy as np
from tflite_runtime.interpreter import Interpreter
from PIL import Image

# ========== 1. 加载模型 ==========
# 创建解释器，加载 tflite 模型
interpreter = Interpreter(model_path="/home/pi/models/mobilenet_v2_1.0_224_quant.tflite")
interpreter.allocate_tensors()  # 分配张量内存

# 获取输入输出张量的信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 打印模型输入输出信息（调试用）
print(f"输入形状: {input_details[0]['shape']}")
print(f"输入类型: {input_details[0]['dtype']}")
print(f"输出形状: {output_details[0]['shape']}")

# ========== 2. 读取并预处理图片 ==========
# 打开图片
img = Image.open("/home/pi/test_images/cat.jpg").convert('RGB')

# 调整大小到模型要求的输入尺寸（224x224）
input_shape = input_details[0]['shape']  # [1, 224, 224, 3]
height = input_shape[1]
width = input_shape[2]
img = img.resize((width, height))

# 转成 numpy 数组，并增加 batch 维度 [224, 224, 3] → [1, 224, 224, 3]
input_data = np.expand_dims(np.array(img), axis=0)

# 如果是量化模型（uint8输入），数据已经是 0-255，直接用
# 如果是 float 模型，需要归一化到 [-1, 1] 或 [0, 1]，具体看模型
input_data = input_data.astype(input_details[0]['dtype'])

# ========== 3. 执行推理 ==========
# 设置输入张量
interpreter.set_tensor(input_details[0]['index'], input_data)

# 运行推理
interpreter.invoke()

# 获取输出结果
output_data = interpreter.get_tensor(output_details[0]['index'])

# ========== 4. 处理输出结果 ==========
# 去掉 batch 维度，得到 [1001] 的分类概率
results = np.squeeze(output_data)

# 加载标签文件
with open("/home/pi/models/image_labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# 找出概率最高的前5个类别
top_k = results.argsort()[-5:][::-1]  # 从大到小排序，取前5个

print("\n=== 推理结果 ===")
for i in top_k:
    # 量化模型的输出是 uint8，需要转换成概率（0-1）
    if output_details[0]['dtype'] == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        prob = (results[i] - zero_point) * scale
    else:
        prob = results[i]
    print(f"  {labels[i]}: {prob:.4f} ({prob*100:.2f}%)")
```

### 5.3 代码逐行解释

| 代码段 | 作用 |
|--------|------|
| `Interpreter(model_path=...)` | 创建 TFLite 解释器，加载模型文件 |
| `allocate_tensors()` | 分配推理所需的内存，**必须调用**，否则推理会报错 |
| `get_input_details()` | 获取模型输入信息（形状、类型、量化参数等） |
| `img.resize(...)` | 把图片缩放到模型要求的尺寸 |
| `np.expand_dims(..., axis=0)` | 增加 batch 维度。模型输入是 `[batch, height, width, channels]` 格式 |
| `set_tensor(...)` | 把预处理好的数据喂给输入张量 |
| `invoke()` | 执行推理，这一步是计算量最大的地方 |
| `get_tensor(...)` | 从输出张量中取出结果 |
| `results.argsort()[-5:][::-1]` | numpy 技巧：找出概率最高的前 5 个索引，从大到小排列 |

### 5.4 安装依赖并运行

```bash
# 安装 PIL 库（如果没有的话）
pip install Pillow

# 运行程序
cd ~/projects/first_inference
python3 inference.py
```

### 5.5 预期输出示例

```
输入形状: [  1 224 224   3]
输入类型: <class 'numpy.uint8'>
输出形状: [   1 1001]

=== 推理结果 ===
  tabby, tabby cat: 0.7820 (78.20%)
  tiger cat: 0.1230 (12.30%)
  Egyptian cat: 0.0540 (5.40%)
  lynx, catamount: 0.0120 (1.20%)
  Siamese cat, Siamese: 0.0080 (0.80%)
```

如果看到类似的输出，恭喜你！第一个 TFLite 推理跑通了！

📝 **小贴士**：第一次运行 `invoke()` 会比较慢（冷启动），可以在正式推理前先跑一次预热。

---

## 六、性能优化技巧

树莓派的算力有限，推理速度是大家最关心的问题。这里给大家介绍 5 个实用的优化技巧，以及预期的性能提升幅度。

### 参考基准

先看一下基准数据（MobileNetV2 224x224，单张图片推理）：

| 设备 | float32 模型 | int8 量化模型 |
|------|-------------|--------------|
| 树莓派 5 (2.4GHz) | ~120ms | ~40ms |
| 树莓派 4B (1.5GHz) | ~250ms | ~90ms |
| 树莓派 Zero 2 W | ~800ms | ~300ms |

> 数据仅供参考，实际速度受模型结构、系统负载、温度等因素影响。

---

### 技巧 1：使用 int8 量化模型（速度提升 2-3 倍）

**原理**：把模型的权重和激活值从 32 位浮点数压缩成 8 位整数，计算量减少 4 倍，内存占用减少 4 倍。

**效果**：
- 推理速度：提升 2-3 倍
- 模型体积：缩小到原来的 1/4
- 精度损失：通常很小（<1%），很多场景下感知不到

**怎么做**：

在训练时或训练后进行量化。使用 TensorFlow Lite Converter：

```python
# 训练后量化（Post-training quantization）
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# 启用 int8 量化
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 提供代表性数据集（用于校准激活值的范围）
def representative_data_gen():
    for input_value in representative_dataset:
        yield [input_value]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_types = [tf.int8]

tflite_model_quant = converter.convert()
```

📝 **小贴士**：如果只是想快速体验量化效果，也可以用动态范围量化（只量化权重，不量化激活），不需要代表性数据集，速度也有 ~2 倍提升。

---

### 技巧 2：降低输入分辨率（速度提升 3-5 倍）

**原理**：CNN 的计算量和输入尺寸的平方成正比。224x224 → 96x96，像素数减少到原来的 1/5，计算量也大幅减少。

| 输入尺寸 | 相对计算量 | 参考速度（Pi 5 int8） |
|---------|-----------|---------------------|
| 224x224 | 1.0x（基准） | ~40ms |
| 160x160 | ~0.5x | ~20ms |
| 128x128 | ~0.3x | ~12ms |
| 96x96 | ~0.2x | ~8ms |

**效果**：
- 推理速度：提升 2-5 倍
- 精度影响：取决于任务难度。简单的分类任务（如猫狗分类）160x160 可能就够了

**怎么做**：

训练时就用小尺寸输入，或者用预训练的小尺寸模型（如 MobileNetV2 1.0 96x96）。

---

### 技巧 3：选择更小的模型（速度提升 2-4 倍）

**原理**：不同的模型结构，计算量差异很大。

| 模型 | 参数量 | 计算量（MACs） | 参考速度（Pi 5 int8） | Top-1 精度 |
|------|--------|--------------|---------------------|-----------|
| MobileNetV3-Large 224 | 5.4M | 219M | ~45ms | 75.2% |
| MobileNetV3-Small 224 | 2.5M | 66M | ~15ms | 67.4% |
| EfficientNet-Lite0 | 4.7M | 412M | ~60ms | 75.1% |
| EfficientNet-Lite2 | 5.3M | 710M | ~100ms | 77.6% |

**效果**：
- 从 Large 换 Small：速度快 3 倍，精度降 ~8%
- 需要根据实际场景权衡速度和精度

**怎么做**：

- 简单任务（分类类别少、物体大）：用 MobileNetV3-Small 或 EfficientNet-Lite0
- 复杂任务：用 MobileNetV3-Large 或 EfficientNet-Lite2
- 实时性要求高的：优先选小模型

---

### 技巧 4：多线程推理（速度提升 1.5-3 倍）

**原理**：TFLite 支持多线程推理，利用树莓派的多核 CPU。

**效果**：
- 4 核 CPU 上开 4 线程：通常能提升 2-3 倍速度
- 不是线性提升，因为有线程同步开销

**怎么做**：

```python
from tflite_runtime.interpreter import Interpreter

# 创建解释器时指定线程数
interpreter = Interpreter(
    model_path="model.tflite",
    num_threads=4  # 设为 CPU 核心数
)
interpreter.allocate_tensors()
```

📝 **小贴士**：最佳线程数通常等于 CPU 核心数。树莓派 4B/5 都是 4 核，设为 4 就行。设太多反而会因为线程切换变慢。

---

### 技巧 5：使用 Coral USB 加速器（TPU，再快 10 倍）

**原理**：Google Coral 是一个 USB 接口的 Edge TPU 芯片，专门用来跑神经网络推理，比 CPU 快很多。

**效果**：
- 推理速度：再快 5-10 倍（相对于 CPU int8）
- MobileNetV2 可以跑到 ~4ms（250 FPS）

**怎么做**：

1. 买一个 Coral USB Accelerator（约 500 元）
2. 安装驱动和运行时：
```bash
# 添加源
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# 安装
sudo apt update
sudo apt install libedgetpu1-std python3-pycoral
```

3. 代码里使用 TPU 解释器：
```python
from pycoral.utils.edgetpu import make_interpreter

interpreter = make_interpreter("model_edgetpu.tflite")
interpreter.allocate_tensors()
```

⚠️ **注意**：模型必须是 int8 量化的，并且要用 edgetpu_compiler 编译一下，生成 `_edgetpu.tflite` 文件才能在 TPU 上运行。

---

### 优化效果总结

把这些优化叠加起来，效果非常可观：

| 优化组合 | 相对速度 | MobileNetV2 推理时间（Pi 5） |
|---------|---------|---------------------------|
| float32 + 单线程（基准） | 1x | ~120ms |
| int8 量化 + 单线程 | 3x | ~40ms |
| int8 + 4线程 | 6x | ~20ms |
| int8 + 小模型(96x96) + 4线程 | 20x | ~6ms |
| Coral TPU（int8 224x224） | 30x | ~4ms |

---

## 七、进阶：实时摄像头推理

静态图片推理不够酷？我们来做一个实时摄像头推理，对着镜头就能识别物体。

### 7.1 安装 OpenCV

```bash
# 安装 headless 版本（不需要 GUI 依赖，体积小）
pip install opencv-python-headless

# 如果需要显示窗口（接了显示器的话），装完整版
# pip install opencv-python
```

⚠️ **注意**：如果在纯命令行（无桌面）环境下运行，一定要装 `opencv-python-headless`，否则会因为缺少 GUI 库而安装失败。

### 7.2 完整代码

创建 `camera_inference.py`：

```python
import time
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# ========== 配置 ==========
MODEL_PATH = "/home/pi/models/mobilenet_v2_1.0_224_quant.tflite"
LABELS_PATH = "/home/pi/models/image_labels.txt"
CAMERA_INDEX = 0  # 0 表示第一个摄像头
NUM_THREADS = 4   # 线程数

# ========== 加载模型 ==========
interpreter = Interpreter(model_path=MODEL_PATH, num_threads=NUM_THREADS)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]

# 加载标签
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ========== 打开摄像头 ==========
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("无法打开摄像头！")
    exit()

# 设置摄像头分辨率（可选，调低可以更快）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# FPS 计算
frame_count = 0
fps = 0
start_time = time.time()

print("摄像头已启动，按 q 退出")

while True:
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        print("读取帧失败")
        break

    # ========== 预处理 ==========
    # BGR → RGB（OpenCV 默认 BGR，模型通常需要 RGB）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 调整大小到模型输入尺寸
    img = cv2.resize(rgb_frame, (width, height))

    # 增加 batch 维度
    input_data = np.expand_dims(img, axis=0).astype(input_details[0]['dtype'])

    # ========== 推理 ==========
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    # ========== 解析结果 ==========
    # 取概率最高的类别
    top_idx = results.argmax()
    
    # 转换为概率值（处理量化模型）
    if output_details[0]['dtype'] == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        prob = (results[top_idx] - zero_point) * scale
    else:
        prob = results[top_idx]

    # ========== 计算 FPS ==========
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        start_time = time.time()

    # ========== 在画面上标注结果 ==========
    # 显示类别和置信度
    label_text = f"{labels[top_idx]}: {prob*100:.1f}%"
    cv2.putText(frame, label_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # 显示 FPS
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 显示画面（如果有显示器的话）
    # cv2.imshow("TFLite Camera Inference", frame)
    
    # 每隔一段时间打印一次结果（无显示器时用）
    if frame_count % 10 == 0:
        print(f"FPS: {fps:.1f} | {labels[top_idx]}: {prob*100:.1f}%")

    # 按 q 退出（需要 GUI 窗口）
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# 清理
cap.release()
cv2.destroyAllWindows()
```

### 7.3 运行

```bash
python3 camera_inference.py
```

按 `Ctrl+C` 停止程序。

### 7.4 提升实时性的小技巧

1. **降低摄像头分辨率**：640x480 就够了，1080p 反而增加预处理时间
2. **跳帧推理**：不是每一帧都推理，比如每 3 帧推理一次，中间帧显示上一次的结果。视觉上看起来还是流畅的。
3. **多线程**：把摄像头采集和推理放到不同的线程里，避免互相阻塞
4. **用更小的模型**：MobileNetV3-Small 96x96 可以做到 30+ FPS

📝 **小贴士**：很多人以为摄像头帧率越高越好，其实对于大多数实时识别场景，5-10 FPS 就足够了。人的反应速度也就 200ms，太快了也没用。

---

## 八、常见问题排查

### Q1：导入 tflite_runtime 失败

**错误信息**：`ModuleNotFoundError: No module named 'tflite_runtime'`

**可能原因及解决方法：**

1. **没安装成功**：重新 `pip install tflite-runtime`，看有没有报错
2. **装到别的 Python 版本里了**：用 `python3 -m pip install tflite-runtime` 安装，用 `python3` 运行
3. **32 位系统找不到包**：确认 `uname -m` 输出 `aarch64`，如果不是就重装 64 位系统
4. **pip 版本太老**：`pip install --upgrade pip` 后再试

---

### Q2："standard_init_linux.go:228" 错误

**错误信息**：`standard_init_linux.go:228: exec user process caused: exec format error`

**原因**：架构不匹配。通常是因为你用了 32 位系统但下载了 64 位的包，或者反过来。

**解决方法**：
```bash
# 确认系统架构
uname -m
# aarch64 = 64位 ARM
# armv7l = 32位 ARM

# 确认 Python 架构
python3 -c "import platform; print(platform.machine())"
```

确保模型和运行时的架构与系统一致。

---

### Q3：摄像头打不开

**错误信息**：`Unable to open camera` 或 `cv2.VideoCapture` 返回 False

**排查步骤：**

1. **确认摄像头已启用**：`sudo raspi-config` → Interface Options → Camera → Enable
2. **确认摄像头连接**：排线有没有插反（蓝色面朝 HDMI 接口方向）
3. **测试摄像头**：
   ```bash
   # 官方摄像头模块
   libcamera-hello

   # USB 摄像头
   ls /dev/video*
   # 应该能看到 /dev/video0 之类的设备
   ```
4. **换个 USB 口试试**：USB 摄像头可能某些口供电不足
5. **检查权限**：`sudo usermod -aG video pi`（把当前用户加入 video 组）

---

### Q4：推理结果全错 / 置信度很低

**现象**：明明图片里是猫，但模型识别成别的东西，或者置信度都很低。

**最可能的原因：预处理不一致**

模型训练时的预处理方式和推理时不一样，导致输入数据分布不对。

**检查清单：**

| 检查项 | 说明 |
|--------|------|
| 颜色通道顺序 | OpenCV 是 BGR，PIL 是 RGB，模型可能要求 RGB |
| 像素值范围 | 是 [0, 255]、[0, 1]、还是 [-1, 1]？ |
| 图像大小 | 是不是模型要求的尺寸？ |
| 归一化方式 | 有没有减均值、除以标准差？ |
| 量化参数 | int8 模型的输入输出 scale/zero_point 对不对？ |

**快速排查方法**：
```python
# 打印输入数据的范围，看看对不对
print(f"输入数据范围: {input_data.min()} ~ {input_data.max()}")
print(f"输入数据类型: {input_data.dtype}")
```

比如 float 模型通常要求输入是 [-1, 1]，如果你传了 [0, 255] 的数据，结果肯定不对。

---

### Q5：推理速度特别慢

**可能原因：**

1. **用的是 float32 模型**：换成 int8 量化模型，速度立竿见影
2. **没有开多线程**：加上 `num_threads=4` 参数
3. **输入分辨率太高**：试试 160x160 或 96x96
4. **温度过高降频**：`vcgencmd measure_temp` 看看温度，超过 80°C 会降频
5. **SD 卡太慢**：用 U3 等级的高速卡，模型加载速度会快很多
6. **用了完整 tensorflow 而不是 tflite-runtime**：tensorflow 推理比 tflite 慢很多

检查温度和频率：
```bash
# 温度
vcgencmd measure_temp

# 当前频率
vcgencmd measure_clock arm
```

---

### Q6：OOM（内存不够）

**错误信息**：`MemoryError` 或进程被 kill 掉

**可能原因及解决方法：**

1. **模型太大**：换小模型，或用量化模型（int8 模型内存占用只有 float32 的 1/4）
2. **一次加载太多模型**：用完的模型及时释放
3. **图片太大**：预处理前就缩小，不要把大图片全读进内存
4. **开启 swap**：临时增加虚拟内存（但会很慢，不推荐长期用）：
   ```bash
   # 增加 1GB swap
   sudo dphys-swapfile swapoff
   sudo nano /etc/dphys-swapfile  # 改 CONF_SWAPSIZE=1024
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```

---

### Q7：温度过高 / 自动降频

**现象**：刚开机推理很快，跑几分钟后变慢了

**原因**：树莓派没有风扇的话，满载几分钟温度就会冲到 80°C 以上，触发过热保护自动降频。

**解决方法：**

1. **加散热片**：最基本的被动散热，能降个 5-10°C
2. **加散热风扇**：主动散热效果最好，能保持在 50-60°C
3. **降低 CPU 频率**：牺牲性能换低温（不推荐）
4. **优化推理**：减少推理频率，让 CPU 有时间休息

```bash
# 查看温度
vcgencmd measure_temp

# 查看是否降频
vcgencmd get_throttled
# 0x0 表示正常，0x50000 表示温度过高降频
```

---

### Q8：SSH 连接不上

**排查步骤：**

1. **树莓派有没有开机**：看指示灯，红灯常亮是电源，绿灯闪烁是读 SD 卡
2. **在同一个网络里吗**：电脑和树莓派要连同一个 WiFi / 局域网
3. **IP 地址对不对**：登录路由器后台看设备列表
4. **hostname 能不能解析**：`ping raspberrypi.local` 试试
5. **SSH 服务有没有开**：接显示器键盘，`sudo systemctl status ssh` 查看

---

### Q9：pip install 很慢 / 超时

**解决方法：**

1. **换国内源**：参考第二章 2.5 节换清华源
2. **使用镜像源安装**：
   ```bash
   pip install tflite-runtime -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
3. **增加超时时间**：
   ```bash
   pip install tflite-runtime --timeout 120
   ```

---

### Q10：模型加载失败 / 格式不对

**错误信息**：`ValueError: Invalid model file` 或 `Didn't find op for builtin opcode`

**可能原因：**

1. **文件损坏**：重新传输一次，用 md5 校验一下
2. **TFLite 版本太旧**：模型是用新版 TensorFlow 转换的，但运行时版本太老，不支持某些算子
3. **模型里有自定义算子**：需要注册自定义算子才能运行

**解决方法：**
- 更新 tflite-runtime 到最新版本
- 用和训练时相同版本的 TensorFlow 转换模型
- 检查模型里的算子是不是 TFLite 支持的：https://www.tensorflow.org/lite/guide/ops_compatibility

---

## 九、项目想法（给学生的灵感）

学完基础的推理后，可以做这些有趣的小项目练手。每个项目一句话说明用到的技术和难度。

### 1. 智能猫脸识别门铃

**难度**：⭐⭐⭐  
**技术**：摄像头 + 人脸/猫脸检测 + 图像分类 + 通知推送

**说明**：门口装一个摄像头，识别到猫咪就拍张照发到你手机上。可以用 MobileNet SSD 做目标检测，找到猫的位置再分类。进阶版可以训练你家猫的专属模型，识别"是我家的猫"还是"别人家的猫"。

---

### 2. 植物健康监测

**难度**：⭐⭐  
**技术**：摄像头 + 图像分类 + 定时采集 + 数据记录

**说明**：对着家里的植物拍照片，识别植物是否健康（有没有黄叶、病虫害）。可以每隔几小时拍一张，记录生长状态。数据集可以自己拍，也可以用公开的植物病害数据集。

---

### 3. 垃圾分类识别

**难度**：⭐⭐⭐  
**技术**：图像分类 + 多类别 + 实物展示

**说明**：对着垃圾拍张照，识别是什么垃圾（可回收/厨余/有害/其他）。非常实用的课程作业项目。可以从 4 大类开始，再细分成更多小类。难点在于垃圾种类多、形态差异大。

---

### 4. 手势控制小灯

**难度**：⭐⭐⭐⭐  
**技术**：手部关键点检测 + 手势分类 + GPIO 控制 LED

**说明**：用摄像头识别手势，比如比"1"开红灯，比"2"开绿灯，握拳全关。可以用 MediaPipe Hands 检测手部关键点，再用一个小模型分类手势。涉及硬件控制，成就感很强。

---

### 5. 数字仪表读数

**难度**：⭐⭐⭐⭐  
**技术**：目标检测 + 数字识别（OCR）+ 数据上传

**说明**：对着家里的电表、水表、温度计拍照片，自动读取数字并记录。先检测表盘区域，再做数字识别。工业场景也很常用，属于"有用"的项目。

---

### 6. 表情识别小玩具

**难度**：⭐⭐⭐  
**技术**：人脸检测 + 表情分类 + 响应式反馈

**说明**：摄像头识别你的表情（开心/难过/惊讶/生气），然后做出反应——比如开心时播放欢快的音乐，难过时显示安慰的话。可以做成一个带屏幕的小摆件，非常有趣。

---

### 7. 更多想法

- **门禁系统**：人脸识别开门
- **停车计数**：检测车位上有没有车
- **跌倒检测**：监控老人是否跌倒（伦理问题要注意）
- **水果成熟度检测**：看颜色判断水果熟没熟
- **车牌识别**：小区门禁用
- **棋盘识别**：自动识别棋局并记录

📝 **小贴士**：做项目的时候，先从最简单的版本开始跑通，再逐步加功能。不要一上来就想做一个"完美的系统"，那样很容易半途而废。先做 MVP（最小可行产品），再迭代优化。

---

## 写在最后

恭喜你看到这里！从刷系统到跑通推理，再到性能优化和项目实战，你已经掌握了树莓派边缘计算的核心技能。

边缘计算是一个很有前景的方向——把 AI 能力放到设备端，低延迟、保护隐私、不需要网络。树莓派是入门边缘计算的最佳选择，便宜、好玩、社区强大。

如果遇到文档里没有提到的问题，欢迎去以下地方找答案：
- 树莓派官方论坛：https://forums.raspberrypi.com/
- TensorFlow Lite 官方文档：https://www.tensorflow.org/lite
- GitHub Issues：搜一下你的错误信息，大概率有人遇到过

祝大家玩得开心，做出有意思的项目！🚀
