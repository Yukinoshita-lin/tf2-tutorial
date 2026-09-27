# 深度学习调试手册（面向初学者）

> 从零基础到独立解决问题，遇到任何报错先翻这本手册。
> 原 FAQ 中的问题已全部整合进对应章节。

---

## 一、先做这 3 件事（遇到问题先自查）

遇到报错先别慌，也别急着问人。按照下面这三步走，80% 的问题你自己就能解决。

### 第 1 步：认真读完报错信息

**现象**：程序崩了，屏幕上一堆红字。

**怎么做**：
- 不要只看最后一行，从报错的最上面开始读；
- 找到写着 `Error` 或 `ValueError`、`TypeError` 的那一行，那就是问题所在；
- 报错信息会告诉你**哪个文件、第几行、出了什么错**，这些都是线索。

**类比**：就像去医院看病，你得先告诉医生哪里疼，而不是只说"我不舒服"。

### 第 2 步：做一个最小可复现的小例子

**现象**：你的代码写了几百行，不知道错在哪。

**怎么做**：
- 新建一个最简单的 `.py` 文件；
- 只保留能触发报错的最少代码（比如只有模型的一层 + 一个假数据）；
- 如果最小例子也报错，说明问题就在这几行里；如果不报错，就逐步加代码，直到复现问题。

**类比**：就像自行车坏了，你先把链子拆下来单独转一转，看看是不是链子的问题。

### 第 3 步：打印数据和模型的形状

**现象**：报错说形状不匹配，但你不知道哪里不匹配。

**怎么做**：
- 在报错的那一行之前，把所有张量的形状都打印出来：
  ```python
  print("x 的形状：", x.shape)
  print("y 的形状：", y.shape)
  print("模型期望输入形状：", model.input_shape)
  ```
- 对比一下，哪边多了一维、哪边数字对不上。

**类比**：就像拼图，你得先看看手里的拼图块是几乘几的，再看看拼图板上的洞是多大。

---

## 二、环境与安装问题

### 问题 1：`ModuleNotFoundError: No module named 'tensorflow'`

**现象**：一运行代码就说找不到 tensorflow 模块。

**常见原因**：
1. 你运行代码用的 Python，和装了 TensorFlow 的 Python 不是同一个（最常见）；
2. 确实还没装 TensorFlow；
3. 虚拟环境没有激活。

**怎么排查**：
```powershell
# 看看当前用的是哪个 python
where python
# 看看这个 python 里有没有装 tensorflow
python -c "import tensorflow; print(tensorflow.__version__)"
```

**解决办法**：

**方法一：用项目自带的虚拟环境（推荐）**

本项目的 TensorFlow 装在 `F:\tf2-env`（Python 3.12.9 + TensorFlow 2.21.0，仅 CPU）。
有三种调用方式：

```powershell
# 方式 1：用转发脚本（最省心）
python scripts\use_tf2_env.py chapters\03_mlp_mnist.py

# 方式 2：直接调用解释器
F:\tf2-env\Scripts\python.exe chapters\03_mlp_mnist.py

# 方式 3：先激活虚拟环境
F:\tf2-env\Scripts\Activate.ps1
# 然后就可以直接 python xxx.py 了
```

> 如果你的 TF 装在别的位置，改 `scripts/use_tf2_env.py` 里的 `DEFAULT_VENV` 就行。

**方法二：自己装一个**

如果你想在自己的环境里装：
```powershell
# 1. 先确认 Python 版本（TF 2.16+ 支持 3.9~3.12）
python --version

# 2. 创建虚拟环境（推荐）
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. 安装（用清华镜像更快）
pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题 2：`pip install tensorflow` 装不上 / 报错

**现象**：pip 安装时报错，或者下载到一半失败。

**常见原因**：
1. Python 版本不兼容（比如 Python 3.14 太新了）；
2. 网络问题，下载太慢超时；
3. 电脑上同时有多个 Python，装错地方了。

**怎么排查**：
```powershell
python --version   # 看版本
pip --version      # 看 pip 属于哪个 Python
```

**解决办法**：

1. **版本不对**：TF 2.16+ 官方支持 Python 3.9–3.12。如果你的 `python` 是 3.13 或 3.14，
   用 `py -3.12 -m venv .venv` 建一个 3.12 的虚拟环境再装。

2. **网络慢**：用清华镜像源：
   ```powershell
   pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

3. **想省时间**：装 CPU-only 版本（更小、更快）：
   ```powershell
   pip install tensorflow-cpu
   ```

4. **还不行**：去 [TensorFlow 官网](https://www.tensorflow.org/install) 找离线 wheel 包手动安装。

### 问题 3：`AttributeError: module 'tensorflow' has no attribute 'keras'`

**现象**：调用 `tf.keras` 时报错说没有这个属性。

**常见原因**：
1. 全局环境里混装了多个版本的 TensorFlow 和独立的 Keras 包，互相打架；
2. 装的是 TensorFlow 1.x，而代码是 TensorFlow 2.x 的写法。

**怎么排查**：
```python
import tensorflow as tf
print(tf.__version__)   # 看看版本号
```

**解决办法**：

最干净的办法是在全新的虚拟环境里重装：

```powershell
# 建一个全新的虚拟环境
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 先卸载干净（防止残留）
python -m pip uninstall -y tensorflow keras tf-nightly keras-nightly

# 重新安装
python -m pip install tensorflow
```

### 问题 4：电脑有显卡，但 TensorFlow 只用 CPU

**现象**：`env_check.py` 显示 `TF GPU devices: []`，明明有 RTX 显卡。

**常见原因**：
TensorFlow ≥ 2.11 在 **Windows 原生环境只提供 CPU 版**（这是官方限制，跟你装什么驱动没关系）。

**怎么排查**：
```powershell
python scripts\env_check.py
# 看看 "TF GPU devices" 那一行
```

**解决办法**：

有三条路可以选：

1. **WSL2 + Ubuntu（推荐）**：在 WSL2 的 Ubuntu 里装 TensorFlow：
   ```bash
   pip install tensorflow[and-cuda]
   ```
   （本机已经装好了 WSL2 Ubuntu，可以直接用）

2. **Colab / Kaggle**：把项目上传到 GitHub，然后在 Colab 里打开，选 GPU 运行时。

3. **TensorFlow-DirectML 插件**：Windows 上通过 DirectML 调用 GPU（第三方维护，可能有兼容性问题）。

> 放心：本项目所有章节在 CPU 上都能完整跑通，GPU 只影响速度，不影响代码写法和学习效果。

### 问题 5：怎么检查环境对不对

**现象**：不确定自己的环境有没有装好。

**解决办法**：运行环境检查脚本：

```powershell
python scripts\env_check.py
```

输出会列出 Python 版本、TensorFlow 版本、GPU 设备等信息。如果所有项都是绿色的 "OK"，说明环境没问题。

### 问题 6：`AttributeError: module 'keras.api._v2.keras' has no attribute ...`

**现象**：调用 Keras 的某个函数时报错说没有这个属性。

**常见原因**：
1. TensorFlow 版本太新或太旧，API 变了；
2. 用了 Keras 3 的写法，但环境里是 Keras 2（或者反过来）。

**怎么排查**：
```python
import tensorflow as tf
print(tf.__version__)   # TF 版本
print(tf.keras.__version__)  # Keras 版本
```

**解决办法**：
- TensorFlow 2.16 及以后默认用 Keras 3，API 有一些变化；
- 查 [官方文档](https://www.tensorflow.org/api_docs/python/tf/keras) 确认你用的函数在当前版本是否存在；
- 本项目的代码已经适配了 TF 2.21 + Keras 3，如果你用的是旧版本 TF，可能需要升级。

### 问题 7：没装 TensorFlow，能跑这个项目吗？

**现象**：还没装 TF，想先看看项目内容。

**回答**：可以。`chapters/01_tensors_autograd.py` 只用 NumPy，不需要 TF；其余章节需要 TF。
建议还是先装好环境再学习，不然跑不了代码光看文字效果会打折扣。

### 问题 8：怎么在 PyCharm / VSCode 里调试

**现象**：想用 IDE 的断点调试功能。

**解决办法**：

- **VSCode**：直接打开 `F:\Tensorflow` 文件夹，打开 `chapters/03_mlp_mnist.py`，在行号左边点一下设断点，按 F5 开始调试。
  注意选对 Python 解释器（`F:\tf2-env\Scripts\python.exe`）。

- **PyCharm**：在 `Settings → Project → Python Interpreter` 里选择解释器
  （`F:\tf2-env\Scripts\python.exe`），再把 `src/` 目录右键标记为 Sources Root。

---

## 三、数据相关问题

### 问题 1：数据下载卡住 / 一直显示 "Downloading data"

**现象**：`model.fit` 一直停在第一个 epoch，或者运行到加载数据的地方就不动了。

**常见原因**：
1. 第一次运行需要下载数据集，网络慢导致卡住；
2. 已经下载了一部分，但文件损坏了，重新下载又失败。

**怎么排查**：
- 看看输出里有没有 `Downloading data from ...` 的字样；
- 如果有，耐心等一会儿（MNIST 约 11MB，CIFAR-10 约 170MB）。

**解决办法**：

1. **耐心等待**：第一次下载数据需要时间，取决于网速。
2. **换镜像源**：如果是 keras 数据集，可以设置环境变量用国内镜像：
   ```powershell
   set KERAS_HOME=F:\Tensorflow\data
   ```
3. **手动下载**：去数据集官网手动下载，放到对应目录里。
   - MNIST 数据默认存在 `~/.keras/datasets/mnist.npz`
   - CIFAR-10 默认存在 `~/.keras/datasets/cifar-10-batches-py/`

### 问题 2：`Unknown image file format. One of JPEG, PNG, GIF, BMP required.`

**现象**：加载图片时报错说格式不对。

**常见原因**：
1. 文件后缀是 `.jpg` 但实际不是图片（比如是个文本文件改了后缀）；
2. 图片文件损坏了；
3. 路径里有非图片文件（比如把 `.db` 文件也读进来了）。

**怎么排查**：
```python
import os
from PIL import Image

img_path = "xxx.jpg"
try:
    img = Image.open(img_path)
    img.verify()  # 验证是不是真图片
    print("图片正常")
except Exception as e:
    print(f"损坏的图片: {img_path}, 错误: {e}")
```

**解决办法**：
- 找出损坏的文件，删掉或替换；
- 加载数据时加个过滤，只读取正确的图片格式：
  ```python
  allowed_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
  image_files = [f for f in all_files if f.lower().endswith(allowed_formats)]
  ```

### 问题 3：数据形状不对 / shape mismatch

**现象**：报错说形状不匹配，比如 `expected shape=(None, 28, 28, 1) but got shape=(None, 28, 28)`。

**常见原因**：
1. 模型期望的输入形状和你喂进去的数据形状不一样（比如多了或少了一个通道维度）；
2. 彩色图当成灰度图用了，或者反过来；
3. 数据被展平了但模型期望的是二维图片。

**怎么排查**：
在喂数据之前打印形状：
```python
print("训练数据形状:", x_train.shape)
print("模型期望输入:", model.input_shape)
```

**解决办法**：

根据需要调整形状：

```python
# 增加通道维度（比如 (60000, 28, 28) → (60000, 28, 28, 1)）
x_train = x_train[..., tf.newaxis]

# 去掉多余维度
x_train = tf.squeeze(x_train, axis=-1)

# 展平成一维（MLP 用）
x_train = x_train.reshape(-1, 28 * 28)
```

### 问题 4：数据归一化做错了，会怎样？

**现象**：loss 一开始就很大，或者训练不动，但不报错。

**常见原因**：
1. 忘了做归一化（像素值还是 0~255）；
2. 归一化的方式不对（比如应该除以 255，你除以了 256）；
3. 训练集做了归一化，但测试集忘了做。

**怎么排查**：
```python
print("数据范围:", x_train.min(), "~", x_train.max())
# 正常应该是 0.0 ~ 1.0 或者 -1.0 ~ 1.0
```

**解决办法**：

```python
# 最常见的归一化：缩放到 0~1
x_train = x_train / 255.0
x_test = x_test / 255.0   # 测试集也要做！

# 或者标准化到均值 0 方差 1
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std   # 注意用训练集的均值和方差
```

**类比**：就像考试打分，如果有的科目满分 100、有的满分 1000，直接加起来比较就不公平。归一化就是把所有科目都换算成满分 100。

### 问题 5：标签格式不对（one-hot vs 稀疏标签）

**现象**：报错 `Shapes (None, 10) and (None, 1) are incompatible`，或者 loss 一直很大。

**常见原因**：
1. 损失函数用错了。如果标签是 `[0, 3, 7]` 这种单个数字（稀疏标签），应该用 `SparseCategoricalCrossentropy`；
2. 如果标签是 `[[1,0,0,...], [0,0,0,1,...], ...]` 这种 one-hot 编码，应该用 `CategoricalCrossentropy`；
3. 搞混了，用反了。

**怎么排查**：
```python
print("标签形状:", y_train.shape)
print("第一个标签:", y_train[0])
# 如果是数字 → SparseCategoricalCrossentropy
# 如果是数组 → CategoricalCrossentropy
```

**解决办法**：

```python
# 情况 1：标签是数字（如 [5, 0, 3, ...]）
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)

# 情况 2：标签是 one-hot（如 [[0,0,...1,...], ...]）
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)
```

### 问题 6：`tf.data` 管道卡住 / 死循环

**现象**：用 `tf.data.Dataset` 构建的数据管道，运行起来一直卡住没输出。

**常见原因**：
1. 数据集无限重复但忘了设 epoch 或 step 数；
2. `prefetch` 或 `batch` 配置有问题；
3. 数据生成器里有死循环。

**怎么排查**：
- 先取一个数据看看能不能正常取到：
  ```python
  for x, y in train_dataset.take(1):
      print("拿到一个 batch，形状：", x.shape, y.shape)
  ```
- 如果这一步都卡住，说明数据集本身有问题。

**解决办法**：

1. 检查有没有 `.repeat()`，如果有，`model.fit` 里必须指定 `steps_per_epoch`；
2. 确保 `batch()` 和 `prefetch()` 都正确配置了；
3. 从最简单的管道开始，一步步加功能，每次都验证一下。

### 问题 7：数据集太小怎么办

**现象**：训练数据太少，模型很快就过拟合了。

**常见原因**：
1. 实际场景中数据收集困难；
2. 数据集划分不合理，训练集太少。

**解决办法**：

1. **数据增强（Data Augmentation）**：对已有图片做旋转、翻转、缩放等操作，"变出"更多数据：
   ```python
   data_augmentation = tf.keras.Sequential([
       tf.keras.layers.RandomFlip("horizontal"),
       tf.keras.layers.RandomRotation(0.1),
       tf.keras.layers.RandomZoom(0.1),
   ])
   ```

2. **迁移学习**：用在大数据集上预训练好的模型，只训最后几层。

3. **合理划分**：训练集、验证集、测试集的比例一般是 6:2:2 或 7:1:2，别让训练集太少。

### 问题 8：数据增强没效果 / 反而更差

**现象**：加了数据增强，准确率反而下降了。

**常见原因**：
1. 增强得太狠了，图片变得面目全非，模型认不出来；
2. 测试集也做了增强（不对！测试集要保持原样）。

**解决办法**：

1. 数据增强**只在训练集上用**，验证集和测试集不要用；
2. 增强的幅度从小到大调，别一下就转 90 度：
   ```python
   # 温和的增强（推荐先从这个开始）
   tf.keras.layers.RandomFlip("horizontal")    # 水平翻转
   tf.keras.layers.RandomRotation(0.05)        # 最多转 5% * 360 = 18度
   tf.keras.layers.RandomZoom(0.05)            # 最多缩放 5%
   ```

---

## 四、模型构建问题

### 问题 1：`Input 0 of layer ... is incompatible with the layer`

**现象**：模型一调用就报错，说输入形状和层不兼容。

**常见原因**：
1. 第一层的 `input_shape` 设错了；
2. 数据形状和 `input_shape` 对不上；
3. 中间某层把形状改了，后面的层接不上。

**怎么排查**：
- 先看 `model.summary()`，每一层的输出形状都列出来了；
- 对比报错信息里的 `expected shape` 和实际输入形状。

**解决办法**：

```python
# 错误示例：输入是 28x28 的图片，但 input_shape 写成了 32x32
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(32, 32)),  # 错了！
    ...
])

# 正确写法：根据数据形状来
# 如果数据是 (None, 28, 28) 的扁平化输入
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # 先展平
    tf.keras.layers.Dense(128, activation='relu'),
    ...
])
```

> 小技巧：用 `model.summary()` 检查每一层的输出形状，确保跟你想的一样。

### 问题 2：输出层单元数和损失函数不匹配

**现象**：二分类问题用了 10 个输出单元，或者多分类用了 1 个输出单元，结果 loss 不对或报错。

**常见原因**：
1. 搞不清二分类和多分类的区别；
2. 输出层的激活函数和损失函数配错对了。

**解决办法**：

记住这个表格：

| 问题类型 | 输出单元数 | 输出激活函数 | 损失函数 |
|---------|-----------|-------------|---------|
| 二分类（是/不是） | 1 | sigmoid | BinaryCrossentropy |
| 多分类（10 选 1） | 类别数（如 10） | softmax | CategoricalCrossentropy |
| 多分类 + 稀疏标签 | 类别数 | softmax | SparseCategoricalCrossentropy |
| 回归（预测数值） | 1 | 不用（或 linear） | MSE / MAE |

```python
# 二分类示例
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', ...)

# 多分类示例（10 类，标签是数字）
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', ...)
```

> 小贴士：如果你不确定，最后一层不加激活函数（即 `from_logits=True`），让损失函数内部处理，一般更稳定。

### 问题 3：模型加了更多层，效果反而更差

**现象**：以为层数越多越厉害，结果加深网络后准确率反而降了。

**常见原因**：
1. 模型太复杂，数据不够，过拟合了；
2. 梯度消失，深层网络学不动；
3. 没有加 BatchNormalization 等技巧。

**怎么排查**：
- 看训练准确率和验证准确率的差距。如果训练准确率很高、验证准确率很低，就是过拟合。

**解决办法**：

1. **先从简单模型开始**：比如 2~3 层的 MLP，确认能跑通、有效果，再慢慢加深；
2. **加 BatchNormalization**：每一层卷积或全连接后面加一个：
   ```python
   model.add(tf.keras.layers.Dense(128))
   model.add(tf.keras.layers.BatchNormalization())
   model.add(tf.keras.layers.Activation('relu'))
   ```
3. **加 Dropout**：防止过拟合：
   ```python
   model.add(tf.keras.layers.Dropout(0.2))  # 随机丢掉 20% 的神经元
   ```
4. **用残差连接（ResNet 思路）**：让深层网络更容易训练。

**类比**：就像学习，不是参考书越多越好。书太多了你消化不了，反而会混乱。适合自己的才是最好的。

### 问题 4：激活函数怎么选

**现象**：不知道该用 ReLU 还是 sigmoid，还是 tanh。

**快速答案**：

| 位置 | 推荐激活函数 | 为什么 |
|-----|-------------|-------|
| 隐藏层 | ReLU（或 LeakyReLU） | 计算快，不容易梯度消失 |
| 二分类输出层 | sigmoid | 输出 0~1 之间的概率 |
| 多分类输出层 | softmax | 所有输出加起来等于 1，像概率分布 |
| 回归输出层 | 不用（linear） | 直接输出数值 |

```python
# 隐藏层标配
tf.keras.layers.Dense(128, activation='relu')

# 二分类输出
tf.keras.layers.Dense(1, activation='sigmoid')

# 多分类输出
tf.keras.layers.Dense(10, activation='softmax')
```

> 为什么隐藏层不用 sigmoid？因为 sigmoid 在输入很大或很小时梯度几乎为 0，深层网络会"学不动"（梯度消失）。

### 问题 5：`model.summary()` 看不懂

**现象**：打印模型总结，但不知道那些数字是什么意思。

**解释**：

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 flatten (Flatten)           (None, 784)               0          ← 展平层，没参数
 dense (Dense)               (None, 128)               100480     ← 784*128 + 128 = 100480
 dense_1 (Dense)             (None, 10)                1290       ← 128*10 + 10 = 1290
=================================================================
Total params: 101,770
Trainable params: 101,770        ← 可训练的参数总数
Non-trainable params: 0          ← 冻结的参数（迁移学习时会有）
_________________________________________________________________
```

- **Output Shape**：这一层输出的形状，`None` 表示 batch 大小不固定；
- **Param #**：这一层有多少个参数（权重 + 偏置）；
- 参数越多，模型越"能记"，但也越容易过拟合、越慢。

### 问题 6：迁移学习时为什么要冻结层

**现象**：用预训练模型做迁移学习，教程里说要先 `base_model.trainable = False`。

**为什么**：

1. **预训练模型已经学会了很多**（比如识别边缘、纹理、形状），这些底层特征不用重新学；
2. **如果不冻结**，随机初始化的分类层会产生很大的梯度，把预训练好的权重"冲坏"；
3. 正确的做法是：先冻结底层，训练分类层 → 然后解冻部分层，用很小的学习率微调。

```python
# 第一步：冻结基础模型，只训分类头
base_model = tf.keras.applications.MobileNetV2(include_top=False, ...)
base_model.trainable = False   # 冻结

# 第二步：训练几轮后，解冻顶层微调
base_model.trainable = True
fine_tune_at = 100  # 解冻第 100 层之后的所有层
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), ...)  # 学习率要小！
```

### 问题 7：模型参数太多 / 显存不够

**现象**：模型太大，加载或训练时 OOM。

**解决办法**：

1. **减少每层的神经元数 / 通道数**：比如从 512 降到 256；
2. **减少层数**：模型简单点；
3. **用更小的输入图片**：比如从 224x224 降到 96x96；
4. **用深度可分离卷积**（MobileNet 思路）：参数少很多；
5. **加梯度检查点**（Gradient Checkpointing）：用时间换空间。

### 问题 8：`No gradients provided for any variable`

**现象**：训练时报错说没有梯度。

**常见原因**：
1. 损失函数的计算没有经过模型的可训练变量；
2. 用了 `tf.stop_gradient()` 或者把张量转成了 numpy 再运算；
3. 自定义损失函数写错了，计算图断了。

**怎么排查**：
- 检查损失函数里的每一步操作，确保都是用 TF 算子算的，没有中途转 numpy。

**解决办法**：

```python
# 错误示例：中间转成了 numpy，梯度就断了
def bad_loss(y_true, y_pred):
    diff = y_pred.numpy() - y_true.numpy()  # 错！转 numpy 就没梯度了
    return tf.reduce_mean(diff ** 2)

# 正确示例：全程用 TF 算子
def good_loss(y_true, y_pred):
    diff = y_pred - y_true
    return tf.reduce_mean(tf.square(diff))
```

---

## 五、训练过程问题（最常见）

这一章是全书最重要的部分。训练就像教学生——有的学生学不会，有的学太死，有的学太快忘得也快。
下面这些问题，每一个深度学习初学者都会遇到。

### 问题 1：loss 一直不下降 / 降得特别慢

**现象**：训练了好几个 epoch，loss 几乎没变化，准确率也跟瞎猜差不多。

**常见原因**：
1. 学习率太小了，步子迈得太小，走半天还在原地；
2. 数据没归一化，导致梯度更新困难；
3. 模型结构有问题（比如全是 sigmoid 激活，梯度消失了）；
4. 标签搞反了 / 数据和标签不对应。

**怎么排查**：

```python
# 1. 看看学习率是多少
print(model.optimizer.learning_rate.numpy())

# 2. 看看数据范围对不对
print(x_train.min(), x_train.max())   # 应该是 0~1 左右

# 3. 看看数据和标签对不对得上
import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.title(f"标签: {y_train[0]}")
plt.show()
```

**解决办法**：

1. **调大学习率**：从 `0.001` 改成 `0.01` 试试；
2. **检查数据归一化**：确保除以了 255；
3. **换激活函数**：隐藏层用 ReLU，别用 sigmoid；
4. **确认标签正确**：可视化几张图，看看图和标签是不是对应的。

**类比**：loss 不下降就像学车时方向盘没动——要么是你没踩油门（学习率太小），要么是方向盘卡住了（数据/模型有问题）。

### 问题 2：loss 震荡 / 来回跳，就是不往下走

**现象**：loss 曲线像过山车一样忽上忽下，整体没怎么下降。

**常见原因**：
1. 学习率太大了，步子迈得太大，来回晃；
2. batch size 太小，每个 batch 的噪声大；
3. 数据本身噪音大，或者标签有错误。

**怎么排查**：
- 把学习率调小 10 倍试试，如果震荡减轻了，说明就是学习率的问题。

**解决办法**：

1. **减小学习率**：从 `0.01` 降到 `0.001`，再不行降到 `0.0001`；
2. **增大 batch size**：从 32 改成 64 或 128（如果显存够的话）；
3. **用学习率衰减**：训练过程中逐步降低学习率：
   ```python
   lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
       initial_learning_rate=0.01,
       decay_steps=1000,
       decay_rate=0.9
   )
   optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
   ```

**类比**：loss 震荡就像下山时步子迈太大——你一脚迈出去直接跨过了山谷，到了对面的山坡上，然后又迈回来，来回晃悠，就是到不了谷底。

### 问题 3：loss 变成 NaN / Inf

**现象**：训练到一半 loss 变成了 `nan`（不是数字），或者 `inf`（无穷大）。

**常见原因**：
1. 学习率太大，参数更新得太猛，数值爆炸了；
2. 数据里有 NaN 或异常值；
3. 损失函数计算时除以了 0，或者 log(0)；
4. 梯度爆炸。

**怎么排查**：

```python
# 1. 检查数据里有没有 NaN
print("有 NaN 吗？", tf.math.is_nan(x_train).numpy().any())

# 2. 看看标签有没有问题
print("标签范围:", y_train.min(), y_train.max())
```

**解决办法**：

1. **降低学习率**（最常见的解决办法）；
2. **检查数据**：确保没有 NaN、Inf 或异常大的值；
3. **给损失函数加 epsilon**：防止 log(0)：
   ```python
   # 自定义损失时加个小常数
   epsilon = 1e-7
   loss = -tf.reduce_mean(y_true * tf.math.log(y_pred + epsilon))
   ```
4. **加梯度裁剪（Gradient Clipping）**：限制梯度的最大值，防止爆炸：
   ```python
   optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
   ```

### 问题 4：训练准确率很高，但验证准确率很低（过拟合）

**现象**：训练准确率 99%，但验证准确率只有 70%，差距很大。loss 曲线训练集一直降，验证集先降后升。

**什么意思**：模型"背下来"了训练集的答案，但遇到新题就不会了。就像学生考试前背了答案，遇到新题就傻眼。

**常见原因**：
1. 模型太复杂，参数太多，记的能力太强；
2. 训练数据太少；
3. 训练时间太长。

**解决办法**（按效果排序）：

1. **数据增强**：增加训练数据的多样性（见第三章问题 7）；
2. **加 Dropout**：
   ```python
   model.add(tf.keras.layers.Dropout(0.3))  # 随机丢掉 30% 的输出
   ```
3. **权重正则化（Weight Decay）**：
   ```python
   model.add(tf.keras.layers.Dense(128, activation='relu',
       kernel_regularizer=tf.keras.regularizers.l2(0.001)))
   ```
4. **早停（Early Stopping）**：验证 loss 不下降了就停止训练：
   ```python
   early_stop = tf.keras.callbacks.EarlyStopping(
       monitor='val_loss', patience=5, restore_best_weights=True)
   model.fit(..., callbacks=[early_stop])
   ```
5. **简化模型**：减少层数或神经元数；
6. **加 Batch Normalization**。

### 问题 5：训练准确率和验证准确率都很低（欠拟合）

**现象**：训练准确率只有 50%，验证准确率也差不多，都很低。

**什么意思**：模型太笨了，连训练集都没学会。就像学生上课没听懂，作业也不会做，考试当然也考不好。

**常见原因**：
1. 模型太简单，容量不够；
2. 学习率太小，学的太慢；
3. 训练时间不够；
4. 特征工程没做好，模型不知道学什么。

**解决办法**：

1. **加深 / 加宽模型**：加层、加神经元、加通道数；
2. **调大学习率**；
3. **多训练几个 epoch**；
4. **换更好的优化器**：比如从 SGD 换成 Adam；
5. **检查数据和标签**：确保数据是对的。

### 问题 6：val_loss 先降后升

**现象**：验证 loss 在前几个 epoch 下降，然后开始上升，但训练 loss 还在一直降。

**什么意思**：这就是**过拟合的信号**。模型开始记住训练集的细节，泛化能力下降了。

**解决办法**：

1. **早停（Early Stopping）**：这是最直接的办法，在验证 loss 开始上升前停下：
   ```python
   early_stop = tf.keras.callbacks.EarlyStopping(
       monitor='val_loss',
       patience=3,        # 连续 3 个 epoch 没改善就停
       restore_best_weights=True  # 恢复到最好的那轮的权重
   )
   ```
2. 其他过拟合解决办法参考问题 4。

### 问题 7：每个 epoch 开始时 loss 突然跳一下

**现象**：每个 epoch 的第一个 batch loss 突然变大，然后又降下去，像锯齿一样。

**常见原因**：
1. 这是正常的！因为每个 epoch 开始时，训练数据被重新打乱（shuffle）了，第一个 batch 可能刚好比较难；
2. 如果你用了 `Dropout` 或 `BatchNormalization`，训练模式和验证模式切换也会让 loss 看起来跳一下。

**怎么判断正不正常**：
- 如果每个 epoch 整体的 loss 是在下降的，只是第一个 batch 高一点，那**完全正常**，不用管；
- 如果整体 loss 不下降，那才是有问题（参考问题 1）。

**解决办法**：不用解决，这是正常现象。看 loss 曲线要看整体趋势，别盯着单个 batch 看。

### 问题 8：准确率卡在某个值不动（比如卡在 10%）

**现象**：训练了很久，准确率一直卡在 10%（或 1/n，n 是类别数）不动，loss 也不降。

**什么意思**：模型等于在瞎猜。10 个类别的话，瞎猜也有 10% 的准确率。

**常见原因**：
1. 学习率太小，模型根本没在学；
2. 数据没归一化，梯度太小；
3. 标签和损失函数不匹配（比如多分类用了二元交叉熵）；
4. 模型结构有严重问题（比如输出层激活函数错了）。

**怎么排查**：

```python
# 看看随机猜的准确率是多少
num_classes = 10
print(f"随机猜的准确率: {1/num_classes * 100:.1f}%")
# 如果你的准确率跟这个差不多，说明模型基本没学
```

**解决办法**：

1. **检查损失函数和标签是否匹配**（第三章问题 5）；
2. **检查输出层激活函数**（第四章问题 2）；
3. **调大学习率**；
4. **确认数据做了归一化**；
5. **确认数据和标签是对应的**（没有 shuffle 时只打乱了 x 没打乱 y）。

### 问题 9：梯度消失 / 梯度爆炸，怎么判断

**现象**：
- **梯度消失**：loss 下降非常慢，深层网络几乎不学，准确率跟随机差不多；
- **梯度爆炸**：loss 变成 NaN 或 Inf，或者参数变得极大。

**怎么检查梯度**：

```python
# 自定义训练循环中打印梯度
with tf.GradientTape() as tape:
    predictions = model(x_batch)
    loss = loss_fn(y_batch, predictions)
grads = tape.gradient(loss, model.trainable_variables)

for g, v in zip(grads, model.trainable_variables):
    print(f"{v.name}: 梯度均值 = {tf.reduce_mean(tf.abs(g)).numpy():.6f}")
```

如果梯度均值特别小（比如 1e-8），就是梯度消失；如果特别大（比如 1e+5），就是梯度爆炸。

**解决办法**：

| 问题 | 解决办法 |
|-----|---------|
| 梯度消失 | 用 ReLU 激活、加 BatchNorm、用残差连接、用预训练模型 |
| 梯度爆炸 | 梯度裁剪 (clipnorm)、降低学习率、加 BatchNorm |

### 问题 10：学习率太大 / 太小分别是什么表现

**怎么判断学习率合不合适**：

| 表现 | 可能原因 | 怎么办 |
|-----|---------|-------|
| loss 一开始就 NaN | 学习率太大 | 调小 10 倍 |
| loss 剧烈震荡，不下降 | 学习率太大 | 调小 3~10 倍 |
| loss 缓慢下降，一直降 | 学习率可能刚好 | 继续观察 |
| loss 降得很快，但很快就不动了 | 学习率可能太大 | 用学习率衰减 |
| loss 几乎不动，像条直线 | 学习率太小 | 调大 10 倍 |
| 准确率卡在随机水平 | 学习率太小或数据问题 | 先调大学习率试试 |

**学习率范围参考**：
- 太大：`> 0.1`（大多数情况）
- 常用：`0.001 ~ 0.01`（Adam 默认 0.001）
- 微调时：`1e-5 ~ 1e-4`
- 太小：`< 1e-6`（除非是微调的最后阶段）

**找合适学习率的技巧**：
从 `1e-7` 开始，每批训练后增大一点（比如乘以 1.1），画 loss-学习率曲线，loss 下降最快的那个点就是好的学习率。

### 问题 11：训练太慢，想快点看到效果

**现象**：电脑性能一般，一个 epoch 要跑好几分钟甚至几十分钟。

**解决办法**：

1. **先少跑几个 epoch**：把 `EPOCHS` 改成 `1` 或 `2`，确认流程能跑通、loss 在下降，再慢慢加；
2. **调小 batch size**（如果是显存不够导致的慢）；
3. **用更小的数据集**：比如先只用 10% 的数据验证思路；
4. **简化模型**：层数少一点、通道数少一点；
5. **用 Colab**：把项目同步到 GitHub，在 Colab 里用 GPU 跑（免费）；
6. **CIFAR-10 是 CPU 上最耗时的章节**，可以留到最后再做。

### 问题 12：`model.fit` 一直停在第一个 epoch

**现象**：开始训练了，但进度条一直卡在 0% 不动。

**常见原因**：
1. 数据下载卡住了（见第三章问题 1）；
2. `tf.data` 管道有死循环（见第三章问题 6）；
3. 数据生成器有问题，卡在了读取文件上。

**怎么排查**：
- 看输出里有没有 `Downloading data from ...`，有就是在下载数据；
- 先单独测试一下数据管道能不能正常出数据。

**解决办法**：参考第三章问题 1 和问题 6。

### 问题 13：训练时内存不足（OOM）

**现象**：报错 `Resource exhausted: OOM when allocating tensor`。

**常见原因**：
1. batch size 太大，一次塞进太多数据；
2. 模型太大，参数太多；
3. 输入图片分辨率太高。

**解决办法**：

1. **调小 `BATCH_SIZE`**：比如从 32 降到 16，再不行降到 8，甚至 4；
2. **减小图片尺寸**：CNN / 迁移学习章节把 `IMAGE_SIZE` 从 160 降到 96；
3. **关闭其他占用内存的程序**：浏览器、视频播放器等；
4. **用梯度累积**：小 batch 模拟大 batch 的效果（进阶技巧）。

---

## 六、评估与预测问题

### 问题 1：测试准确率和验证准确率差很多

**现象**：训练时验证准确率有 90%，但最后在测试集上只有 80%。

**常见原因**：
1. 验证集被"过拟合"了——你反复根据验证集调参，模型间接看到了验证集；
2. 测试集的数据分布和训练/验证集不一样（领域偏移）；
3. 测试集的预处理方式和训练时不一样。

**怎么排查**：
- 检查测试集的预处理是否和验证集完全一致；
- 看看测试集的图片是不是差别很大（比如训练集都是白天的图，测试集都是晚上的）。

**解决办法**：

1. **确保测试集的预处理和验证集完全一样**：归一化、尺寸调整都要相同；
2. **不要用测试集调参**：测试集只能用一次，用来做最终评估；
3. **如果是领域偏移**：考虑领域自适应或者收集更多样的训练数据。

### 问题 2：预测结果全是同一类

**现象**：模型对所有输入都预测成同一个类别。

**常见原因**：
1. 训练数据类别不平衡，某一类占绝大多数；
2. 模型没学好，退化了（比如学习率太大导致的）；
3. 输出层激活函数和损失函数不匹配。

**怎么排查**：
```python
# 看看训练集各类的数量
import numpy as np
unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))
```

**解决办法**：

1. **类别不平衡**：
   - 过采样少数类（重复一些样本）；
   - 欠采样多数类（去掉一些样本）；
   - 给损失函数加类别权重：
     ```python
     class_weight = {0: 1.0, 1: 5.0, 2: 5.0}  # 少数类权重更大
     model.fit(..., class_weight=class_weight)
     ```

2. **检查模型是否真的在学习**（参考第五章问题 1 和问题 8）。

### 问题 3：混淆矩阵看不懂

**现象**：画出来了混淆矩阵，但不知道怎么看。

**解释**：

混淆矩阵是一个 N×N 的表格（N 是类别数），用来展示模型预测的对错情况：

- **行**：真实标签（实际是什么）
- **列**：预测标签（模型认为是什么）
- **对角线上的数字**：预测对了的数量
- **对角线以外的数字**：预测错了的数量

举个例子（3 分类）：

```
          预测为猫  预测为狗  预测为鸟
真实是猫     90       5        5
真实是狗      8      85        7
真实是鸟      3       2       95
```

- 猫有 90 张被正确识别，5 张被认成狗，5 张被认成鸟；
- 模型最容易把狗认成猫（8 个错例）。

**怎么看**：
- 对角线越亮（数字越大）越好，说明预测准确；
- 哪一列（行）以外的地方特别亮，说明模型在那两类上容易搞混。

### 问题 4：精确率、召回率、F1 都是啥

**现象**：看到这些指标不知道什么意思，也不知道该看哪个。

**大白话解释**：

假设模型是个"打假人"，要识别假钞：

| 指标 | 大白话 | 公式 | 什么时候重要 |
|-----|-------|------|-------------|
| 准确率 (Accuracy) | 整体判断对的比例 | 对的 / 全部 | 类别均衡时 |
| 精确率 (Precision) | 说它是假的，有多少真的是假的 | 真猜中 / 猜的全部假货 | 怕冤枉好人（误报代价高） |
| 召回率 (Recall) | 真正的假钞里，你查出了多少 | 真猜中 / 全部真假货 | 怕漏掉坏人（漏报代价高） |
| F1 分数 | 精确率和召回率的调和平均 | 2*P*R/(P+R) | 两者都重要时 |

**举例子**：
- 癌症筛查：召回率更重要（宁可误报，不能漏诊）；
- 垃圾邮件过滤：精确率更重要（宁可漏拦，不能把正常邮件当垃圾）。

**代码示例**：
```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=class_names))
```

### 问题 5：怎么判断模型是"好"还是"坏"

**现象**：准确率 85%，不知道这个成绩算好还是坏。

**判断方法**：

1. **跟随机基线比**：10 分类随机猜是 10%，如果你的模型是 85%，说明还不错；
2. **跟人类水平比**：如果人类能做到 98%，那 85% 还有提升空间；
3. **跟简单模型比**：比如用逻辑回归能做到 80%，你的 CNN 做到 85%，提升不算大；
4. **看实际需求**：如果应用场景只需要 80% 就够用了，那 85% 就挺好。

**一般规律**：
- < 随机水平：模型完全没学好，肯定哪里错了；
- 比随机好一点：模型学到了一点东西，但还不够；
- 接近人类水平：做得不错了；
- 超过人类水平：要么是任务简单，要么你真的很厉害。

### 问题 6：预测时输入形状不对

**现象**：训练好的模型，拿来预测单张图片时报错。

**常见原因**：
1. 模型期望的输入是 `(batch, height, width, channels)` 四维，但你只传了三维的单张图片；
2. 图片尺寸和训练时不一样；
3. 忘了做归一化。

**解决办法**：

```python
# 假设模型输入是 (None, 28, 28, 1)

# 单张图片预测：要加 batch 维度
img = x_test[0]                  # 形状 (28, 28)
img = img / 255.0                # 归一化！别忘了
img = tf.expand_dims(img, axis=0)  # 变成 (1, 28, 28)
img = tf.expand_dims(img, axis=-1) # 变成 (1, 28, 28, 1)

prediction = model(img)
predicted_class = tf.argmax(prediction, axis=1).numpy()[0]
print(f"预测结果：第 {predicted_class} 类")
```

> 小技巧：也可以用 `model.predict(img)`，它会自动处理 batch 维度，但输入还是要对。

---

## 七、保存与部署问题

### 问题 1：模型保存了但加载失败

**现象**：用 `model.save()` 保存了，但 `tf.keras.models.load_model()` 加载时报错。

**常见原因**：
1. 保存和加载的 TensorFlow / Keras 版本不一样；
2. 模型里有自定义层或自定义损失函数，加载时不知道怎么重建；
3. 保存的文件损坏了。

**怎么排查**：
- 看看报错信息里有没有提到 "custom objects" 或版本不兼容。

**解决办法**：

1. **有自定义层/损失函数时**，加载时要声明：
   ```python
   model = tf.keras.models.load_model(
       'my_model.keras',
       custom_objects={'MyLayer': MyLayer, 'my_loss': my_loss}
   )
   ```

2. **版本问题**：尽量保证保存和加载的 TF 大版本一致。
   Keras 3（TF 2.16+）的 `.keras` 格式是新版本，旧版本 TF 可能读不了。

3. **用 SavedModel 格式**（更通用，跨版本兼容好）：
   ```python
   # 保存
   model.export('saved_model')  # Keras 3 写法
   # 加载
   model = tf.saved_model.load('saved_model')
   ```

### 问题 2：`.keras`、`.h5` 和 SavedModel 有什么区别

**现象**：不知道该用哪种格式保存模型。

**对比**：

| 格式 | 后缀 | 特点 | 适用场景 |
|-----|------|------|---------|
| Keras V3 | `.keras` | Keras 3 推荐格式，zip 打包，包含架构+权重+训练配置 | 平时训练保存 checkpoint |
| HDF5 | `.h5` | 旧版 Keras 格式，单文件 | 兼容旧代码时用 |
| SavedModel | 文件夹 | TF 通用格式，包含计算图和权重，可用于部署 | 部署到生产环境 |

**本项目中的产物位置**：
- Keras 格式：`models/chapter08_mnist_mlp.keras`
- SavedModel 格式：`models/saved/chapter08_mnist_mlp/`
- TFLite 格式：`models/chapter08_mnist_mlp.tflite`

> 第 8 章脚本已内置 Keras 3 和旧版 TF 的两种分支，不需要手工改代码。

### 问题 3：TFLite 转换失败

**现象**：想把模型转成 TFLite 格式部署到手机/嵌入式设备，但转换报错。

**常见原因**：
1. 模型里有 TFLite 不支持的算子；
2. SavedModel 导出有问题；
3. 量化时出错。

**怎么排查**：
```python
# 先确认 SavedModel 能正常加载和推理
model = tf.saved_model.load('saved_model')
print(list(model.signatures.keys()))  # 看看有哪些签名
```

**解决办法**：

1. **确保先正确导出 SavedModel**：
   ```python
   # Keras 3 写法
   model.export('saved_model_dir')

   # 旧版 TF 写法
   tf.saved_model.save(model, 'saved_model_dir')
   ```

2. **转换 TFLite**：
   ```python
   converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_dir')
   tflite_model = converter.convert()
   with open('model.tflite', 'wb') as f:
       f.write(tflite_model)
   ```

3. **如果有不支持的算子**：
   - 试试选择 TF 算子的 fallback：
     ```python
     converter.target_spec.supported_ops = [
         tf.lite.OpsSet.TFLITE_BUILTINS,
         tf.lite.OpsSet.SELECT_TF_OPS
     ]
     ```
   - 或者改模型，把不支持的层换成支持的。

### 问题 4：怎么把训练结果导出成图表

**现象**：训练完了，想把 loss 曲线、混淆矩阵等保存下来写报告。

**解决办法**：

本项目每个章节都会自动往 `figures/` 文件夹里保存训练曲线和混淆矩阵图片。
你可以直接把 `figures/*.png` 复制到 Markdown 文档或 PPT 里。

如果你想自己画：

```python
import matplotlib.pyplot as plt

# 画 loss 曲线
plt.plot(history.history['loss'], label='训练 loss')
plt.plot(history.history['val_loss'], label='验证 loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('figures/loss_curve.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 问题 5：模型太大，怎么变小（轻量化）

**现象**：模型有几百 MB，想部署到手机或嵌入式设备上放不下。

**解决办法**（从易到难）：

1. **训练后量化（Post-training quantization）**：最简单，不用重新训练：
   ```python
   converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   tflite_quant_model = converter.convert()
   ```
   通常能把模型缩小 4 倍，速度也更快，准确率损失很小。

2. **用更轻量的网络结构**：
   - 比如用 MobileNet、EfficientNet-Lite 代替 ResNet50；
   - 用深度可分离卷积代替普通卷积。

3. **剪枝（Pruning）**：去掉不重要的权重，需要用 TensorFlow Model Optimization Toolkit。

4. **知识蒸馏（Knowledge Distillation）**：用大模型"教"小模型，进阶技巧。

---

## 八、性能优化（CPU/GPU 加速）

### 问题 1：OOM（内存/显存不足）

**现象**：报错 `Resource exhausted: OOM when allocating tensor`。

**常见原因**：
1. batch size 太大；
2. 模型太大，参数太多；
3. 输入图片分辨率太高；
4. 同时开了其他吃内存的程序。

**解决办法**：

1. **调小 BATCH_SIZE**：32 → 16 → 8 → 4 → 2 → 1，一步步试；
2. **减小图片尺寸**：比如 224 → 160 → 96；
3. **关闭浏览器、视频播放器等大内存应用**；
4. **用混合精度训练**（见问题 6）；
5. **梯度检查点**（Gradient Checkpointing）：用计算时间换显存空间。

### 问题 2：训练太慢，想加速

**现象**：一个 epoch 要跑几十分钟，等得着急。

**解决办法**（从简单到复杂）：

1. **用 GPU**：GPU 训练通常比 CPU 快 10~100 倍（见第二章问题 4）；
2. **数据管道优化**：用 `tf.data` 的 `cache()` 和 `prefetch()`：
   ```python
   dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)
   ```
3. **用混合精度训练**（见问题 6）；
4. **增大 batch size**（如果显存够的话），GPU 利用率更高；
5. **减少训练数据**：先在小数据集上调参，确认可行了再用全量数据。

### 问题 3：怎么知道 GPU 有没有在工作

**现象**：不确定代码是不是真的在用 GPU 跑。

**怎么检查**：

```python
import tensorflow as tf
print("GPU 设备:", tf.config.list_physical_devices('GPU'))
# 如果输出有 GPU 设备，说明 TF 能看到 GPU
```

也可以在训练时看任务管理器（Windows）或 `nvidia-smi`（Linux）：
- Windows：任务管理器 → 性能 → GPU → 看"CUDA"或"3D"的利用率是不是上去了；
- Linux：`watch -n 1 nvidia-smi`，看 GPU 利用率和显存占用。

> 注意：Windows 原生 TF 2.11+ 只有 CPU 版，GPU 利用率当然是 0。这是正常的，不是你没装好。

### 问题 4：batch size 大好还是小好

**现象**：不知道 batch size 设多少合适。

**对比**：

| | batch size 大 | batch size 小 |
|---|-------------|-------------|
| 训练速度 | 更快（GPU 利用率高） | 更慢 |
| 显存占用 | 更大 | 更小 |
| loss 曲线 | 更平滑 | 更震荡 |
| 泛化能力 | 可能稍差 | 可能更好（噪声有正则化效果） |

**建议**：
- 先设成 32 或 64，看看显存够不够；
- 如果显存还剩很多，可以调大到 128 或 256，加快训练；
- 如果 OOM 了，就往小调；
- 最终选择一个**显存不爆、训练速度可以接受**的值。

### 问题 5：`cache()` 和 `prefetch()` 有什么用

**现象**：看到代码里有 `.cache().prefetch(tf.data.AUTOTUNE)`，不知道是干嘛的。

**解释**：

- **`cache()`**：把数据缓存到内存（或文件）里，第一个 epoch 读完数据后，后面的 epoch 直接从内存里拿，不用再读硬盘了。适合数据能完全放进内存的情况。

- **`prefetch()`**：让 CPU 在 GPU 训练当前 batch 的时候，提前准备好下一个 batch。这样 GPU 不用等数据，一直有活干。
  `tf.data.AUTOTUNE` 让 TF 自动决定预取多少个。

**类比**：
- `cache()` 就像把课本复印一份放在桌上，不用每次都去书架拿；
- `prefetch()` 就像吃饭时，你在吃第一口，服务员已经把第二口准备好了，你一口接一口不用等。

### 问题 6：混合精度训练是什么

**现象**：听说混合精度能加速，但不知道是什么。

**解释**：

通常模型参数是用 32 位浮点数（float32）存的。混合精度训练就是一部分用 16 位（float16），一部分还用 32 位，这样：
- 显存占用少一半左右；
- GPU 计算速度更快（尤其是支持 Tensor Core 的显卡）；
- 准确率几乎不受影响。

**怎么用**（Keras 3 写法）：
```python
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

> 注意：
> 1. 只有 GPU 上混合精度才有明显加速，CPU 上效果不大；
> 2. 输出层最好保持 float32，以保证数值稳定性。

---

## 九、常见报错速查表

遇到报错先来这里找。按报错信息的关键字搜索，找到对应的解决方法。

| 报错信息关键字 | 中文解释 | 最可能的原因 | 解决方法 |
|--------------|---------|-------------|---------|
| `Input 0 of layer ... is incompatible with the layer` | 输入形状和层的期望不匹配 | 模型 `input_shape` 设错了，或数据形状不对 | 打印 `x.shape` 和 `model.input_shape` 对比，调整数据形状或模型输入 |
| `ValueError: Shapes (None, 10) and (None, 1) are incompatible` | 标签形状和输出形状不匹配 | 损失函数和标签格式不对应（稀疏标签 vs one-hot） | 标签是数字用 `SparseCategoricalCrossentropy`，标签是 one-hot 用 `CategoricalCrossentropy` |
| `Resource exhausted: OOM when allocating tensor` | 显存/内存不够用了 | batch size 太大，或模型太大 | 调小 batch size，减小图片尺寸，关闭其他占用内存的程序 |
| `WARNING:tensorflow:AutoGraph could not transform` | AutoGraph 转换失败（警告，不是错误） | 自定义函数里有 AutoGraph 不支持的语法 | 一般不影响运行。如果在意，把复杂的 Python 逻辑改成 TF 算子，或加 `@tf.autograph.experimental.do_not_convert` |
| `ModuleNotFoundError: No module named 'tensorflow'` | 找不到 tensorflow 模块 | 环境不对，TF 装在另一个 Python 里，或根本没装 | 用项目虚拟环境 `F:\tf2-env`，或重新安装 TF |
| `AttributeError: module 'tensorflow' has no attribute 'keras'` | tensorflow 模块里没有 keras 属性 | 版本混乱，或装的是 TF 1.x | 新建干净的虚拟环境重装 TF |
| `AttributeError: module 'keras.api._v2.keras' has no attribute ...` | Keras 里没有某个属性/函数 | TF/Keras 版本不对，API 变了 | 查官方文档确认 API，升级或降级 TF 版本 |
| `loss: nan` | 损失变成了非数字 | 学习率太大、数据有 NaN、损失函数计算 log(0) | 降低学习率，检查数据，给损失加 epsilon，加梯度裁剪 |
| `WARNING:tensorflow:Model was constructed with shape ... but it was called on an input with incompatible shape` | 模型构建时的输入形状和实际调用的形状不一样 | 输入数据形状和 `input_shape` 不一致 | 检查数据形状是否正确，可能多了或少了维度 |
| `Unknown image file format. One of JPEG, PNG, GIF, BMP required.` | 未知的图片格式 | 图片文件损坏，或路径里混入了非图片文件 | 找出并删除损坏的文件，加载时过滤文件后缀 |
| `No gradients provided for any variable` | 没有可训练变量的梯度 | 损失计算没经过模型变量，或中途转 numpy 断了计算图 | 确保损失函数全程用 TF 算子，不要 `.numpy()` |
| `Attempting to use uninitialized value` | 尝试使用未初始化的变量 | 变量没初始化就用了（多见于 TF 1.x 风格代码或自定义训练） | 确保变量已经创建并初始化，或先跑一次前向传播 |
| `ValueError: logits and labels must be broadcastable` | 输出和标签形状不能广播 | 输出维度和类别数不匹配 | 检查输出层单元数是否等于类别数，检查标签形状 |
| `InvalidArgumentError: Incompatible shapes` | 张量形状不兼容，无法运算 | 两个张量做运算时形状对不上 | 打印所有相关张量的形状，找出不匹配的地方 |
| `TypeError: '>' not supported between instances of 'Tensor' and 'int'` | 张量和数字直接比较报错 | 在 `@tf.function` 里用了 Python 原生比较 | 用 `tf.cond` 或先 `.numpy()` 取出来再比较（非图模式下） |
| `WARNING:tensorflow:Gradients do not exist for variables` | 某些变量没有梯度 | 损失计算没有经过这些变量 | 检查损失函数是否用到了这些变量，可能是代码逻辑问题 |
| `tf.function-decorated function tried to create variables on non-first call` | tf.function 在非首次调用时创建变量 | 函数内部定义了变量，每次调用都想新建 | 把变量定义移到函数外面，或用 `tf.Variable` 在外部创建 |
| `ValueError: y_true and y_pred have different number of output` | 真实标签和预测标签的输出数量不一样 | 模型输出维度和标签维度不匹配 | 检查模型输出层的单元数和标签的形状是否一致 |

---

## 十、去哪里找更多帮助

如果这本手册里没找到你的问题，可以试试下面这些渠道。

### 1. 官方文档

**TensorFlow 官方文档**：<https://www.tensorflow.org/api_docs/python/tf>

- 最权威、最准确的参考；
- 不确定某个函数怎么用时，直接搜函数名；
- 右上角可以切换版本，注意看你用的是哪个版本。

**Keras 官方文档**：<https://keras.io/api/>

- Keras 3 的文档，比 TF 官网的 Keras 部分更详细；
- 有很多示例代码。

### 2. GitHub Issues

- **搜索已有 Issue**：先搜一下，大概率你遇到的问题别人也遇到过；
- **提 Issue 时**要包含：
  1. 你用的 TF 版本、Python 版本、操作系统；
  2. 完整的报错信息（不是最后一行，是全部）；
  3. 能复现问题的最小代码；
  4. 你已经尝试过的解决方法。

### 3. Stack Overflow

- 网址：<https://stackoverflow.com/>
- 搜问题时加上 `[tensorflow]` 或 `[keras]` 标签；
- 提问前先搜有没有人问过；
- 提问技巧和 GitHub Issue 差不多，信息给得越全，越容易得到回答。

### 4. 中文社区

- **知乎**：搜 "TensorFlow 报错" 之类的关键词；
- **CSDN / 掘金**：很多人会写踩坑记录；
- **B站**：有很多 TF 入门视频教程。

### 5. 本项目的学习资源

- **学习手册**：`docs/learning_handbook_zh.md`，第 9 节有"调试速查"；
- **代码示例**：`chapters/` 目录下每个章节都是完整可运行的例子；
- **环境检查**：`python scripts\env_check.py`，先确认环境没问题。

---

> 最后想说：调试是深度学习的常态，不是例外。哪怕是做了多年的老手，也会天天遇到报错。
> 重要的不是不犯错，而是学会一套系统的排查方法，一步步把问题找出来。
> 祝你学习顺利！

