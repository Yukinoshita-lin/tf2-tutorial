# TensorFlow 2 教学大纲（teaching/chapter_outline.md）

> 📍 学生端的完整知识地图见 `docs/knowledge_map.md`（主线故事、依赖图、三条学习路径、过关标准）；本页是给老师/作者使用的一页纸大纲。

> 一页可打印的版本，方便做课前讲义。

## 章节地图（学习路径）

```
[零依赖]  01 张量 / 自动求导 / NumPy
   ↓
[Keras 入门]
   ↓ 02 线性回归（Sequential / compile / fit）
   ↓
[视觉]
   ↓ 03 MLP → MNIST（flatten + dense）
   ↓ 04 CNN → CIFAR-10（conv / pool / dropout）
   ↓
[文本]
   ↓ 05 IMDB 文本分类（one-hot + dense）
   ↓
[迁移]
   ↓ 06 MobileNetV2 → tf_flowers（冻结 → 微调）
   ↓
[训练工程]
   ↓ 07 回调 / TensorBoard / 早停 / 检查点
   ↓ 08 模型保存（.keras / SavedModel / .tflite）
   ↓
[综合项目]
   ↓ 09 图像分类系统全流程（数据→增强→迁移学习→评估→导出）
   ↓
[部署出口]
   ↓ 10 边缘计算与树莓派部署（TFLite / 量化 / 树莓派 / Coral TPU）
```

## 章节详情表

| 章节 | 标题 | 核心知识点 | 对应脚本 | 预计学时 | 难度 |
| --- | --- | --- | --- | --- | --- |
| 01 | 张量与自动求导 | 张量操作、NumPy 广播、矩阵乘法、手写梯度、GradientTape 自动求导 | chapters/01_tensors_autograd.py | 1 小时 | ★☆☆☆☆ 入门 |
| 02 | 线性回归 | Sequential 模型、compile、fit、线性回归、损失与优化器 | chapters/02_linear_regression.py | 1 小时 | ★☆☆☆☆ 入门 |
| 03 | MLP 与 MNIST | Flatten、Dense、多分类损失、验证集、MNIST 手写数字识别 | chapters/03_mlp_mnist.py | 1-2 小时 | ★★☆☆☆ 初级 |
| 04 | CNN 与 CIFAR-10 | Conv2D、MaxPool2D、Dropout、数据增强、CIFAR-10 图像分类 | chapters/04_cnn_cifar10.py | 2 小时 | ★★☆☆☆ 初级 |
| 05 | IMDB 文本分类 | one-hot 文本向量化、二分类损失、Embedding 变体对比 | chapters/05_text_imdb.py | 1-2 小时 | ★★☆☆☆ 初级 |
| 06 | 迁移学习 | MobileNetV2、冻结、微调、tf_flowers 花朵分类 | chapters/06_transfer_learning.py | 2 小时 | ★★★☆☆ 中级 |
| 07 | 训练工程 | TensorBoard、ModelCheckpoint、EarlyStopping、LambdaCallback 回调 | chapters/07_callbacks_tensorboard.py | 1-2 小时 | ★★★☆☆ 中级 |
| 08 | 模型保存与导出 | .keras / SavedModel / .tflite 三种保存格式 | chapters/08_save_and_export.py | 1 小时 | ★★☆☆☆ 初级 |
| 09 | 图像分类系统全流程 | 数据增强、迁移学习、评估、混淆矩阵、每类准确率、错例分析 | chapters/09_capstone_image_classifier.py | 2-3 小时 | ★★★☆☆ 中级 |
| 10 | 边缘计算与树莓派部署 | 边缘计算、TFLite、模型量化、量化感知训练、推理基准测试、树莓派部署、Coral TPU 加速 | chapters/10_edge_raspberry_pi.py | 1-2 小时 | ★★☆☆☆ 中级工程 |

## 关键概念覆盖矩阵

| 概念 | 出现章节 |
| --- | --- |
| 张量 / NumPy 广播 / 矩阵乘 | 01 |
| 手写梯度（链式法则） | 01 |
| `tf.keras.Sequential` / `Dense` | 02, 03 |
| `compile` + `loss` + `optimizer` | 02, 03, 04, 05 |
| 多类 vs 二分类损失 | 03, 04, 05 |
| `validation_split` / `validation_data` | 02, 03, 04, 07 |
| `Conv2D` / `MaxPool2D` / `Dropout` | 04 |
| 文本向量化（one-hot；Embedding 变体在作业中对比） | 05 |
| 预训练模型 + 冻结 / 微调 | 06 |
| 回调：`TensorBoard` / `ModelCheckpoint` / `EarlyStopping` / `LambdaCallback` | 07 |
| 三种保存格式 | 08 |
| 完整 pipeline：数据增强 + 迁移学习 + 评估 + 导出 | 09 |
| 混淆矩阵 / 每类准确率 / 错例分析 | 09 |
| 边缘计算（入门 / 应用 / 深入） | 08, 09, 10 |

## 推荐作业

1. 把第 1 章 NumPy 版本的"手写梯度"改写成 `tf.GradientTape` 版本；
2. 在第 4 章 CNN 上加入 `tf.keras.layers.RandomFlip` / `RandomTranslation`，观察 val_acc 变化；
3. 在第 6 章用 `ResNet50` 替换 `MobileNetV2`，对比参数量与精度；
4. 第 8 章导出的 `.tflite` 用 TFLite Interpreter 在 Python 中跑一次推理，对比延迟。
5. 第 9 章综合项目：把 MobileNetV2 换成 ResNet50，对比参数量、准确率和训练速度。
6. 把花朵分类模型部署到树莓派上，测试实际 FPS，并与 PC 端推理速度对比。
7. 尝试 int8 全量化（训练后量化或量化感知训练），对比量化前后的精度损失和速度提升。
8. 用树莓派摄像头做一个实时物体识别小装置（如有 Coral TPU 加速器，对比开启前后的 FPS 差异）。


## 进阶篇（主线毕业之后）

| 章 | 主题 | 代码入口 |
| --- | --- | --- |
| 进阶 11 | tf.data 数据管道 | `chapters/11_tfdata_pipeline.py` |
| 进阶 12 | 序列建模 RNN/LSTM | `chapters/12_rnn_timeseries.py` |
| 进阶 13 | 手写注意力/Transformer | `chapters/13_attention_transformer.py` |
| 进阶 14 | 生成模型 AE+GAN | `chapters/14_autoencoder_gan.py` |
| 进阶 15 | 自定义训练与性能 | `chapters/15_custom_training.py` |
| 进阶 16 | 毕业项目: 文本分类五课合一 | `chapters/16_text_capstone.py` |

每章配套 5 板块教学结构（定位/为什么/反例/过关自测/延伸阅读），
学期制安排见 `docs/study_plan_semester.md`。


> 手册附录：A 调试速查 / B 延伸阅读 / **C 反例索引表** / **D 版本迁移速查** /
> **E 数学补充** / **F 术语表** / **G FAQ 速答** / **H 评估指标指南** / **I 进阶方向速览**。
