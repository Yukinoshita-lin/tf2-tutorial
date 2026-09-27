# notebooks/

每个 `.ipynb` 都是 `chapters/` 对应脚本的薄包装：一个 Markdown 单元格解释章节，
一个代码单元格用 `runpy` 直接执行章节脚本，因此不会出现"代码与 notebook 两处维护"。

| 笔记本 | 对应章节 | 数据集 / 依赖 |
| --- | --- | --- |
| `00_ml_basics.ipynb` | 第 0 章 | 仅 NumPy + matplotlib，零 TF 依赖 |
| `01_tensors_autograd.ipynb` | 第 1 章 | 仅 NumPy，零 TF 依赖 |
| `02_linear_regression.ipynb` | 第 2 章 | 合成数据 |
| `03_mlp_mnist.ipynb` | 第 3 章 | MNIST |
| `04_cnn_cifar10.ipynb` | 第 4 章 | CIFAR-10（CPU 最耗时） |
| `05_text_imdb.ipynb` | 第 5 章 | IMDB |
| `06_transfer_learning.ipynb` | 第 6 章 | tf_flowers（需下载权重） |
| `07_callbacks_tensorboard.ipynb` | 第 7 章 | MNIST |
| `08_save_and_export.ipynb` | 第 8 章 | MNIST |
| `09_capstone_image_classifier.ipynb` | 第 9 章 | tf_flowers（需下载权重） |
| `10_edge_raspberry_pi.ipynb` | 第 10 章 | tf_flowers（需下载权重） |

## 使用建议

- 本地：在项目根目录执行 `python teaching\build_notebooks.py` 重新生成；
  然后 `jupyter notebook notebooks`（内核工作目录会落在 `notebooks/`，
  与包装单元格中的 `../chapters/...` 相对路径一致）。
- 云端：把整个 `F:\Tensorflow` 同步到 GitHub / Colab 后再运行，保持目录结构不变。
- 章节脚本通过 `runpy` 直接运行，不要求 `pip install -e .`。
