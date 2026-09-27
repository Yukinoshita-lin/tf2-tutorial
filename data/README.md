# data/

数据集目录。**默认不存放任何数据**：`tensorflow.keras.datasets` 会把数据缓存在用户目录
(如 `~/.keras/datasets`)；`tensorflow_datasets` 的下载由 `TFDS_DATA_DIR` 环境变量指向
`data/tfds`（见 `chapters/06_transfer_learning.py`）。

子目录约定：

| 目录 | 内容 |
| --- | --- |
| `raw/` | 手动下载的离线副本 (`.npz`，供无网环境使用) |
| `processed/` | 经过预处理的 NumPy 数组 |

如果需要把数据集缓存到项目目录内、离线也能跑，可执行：

```powershell
python scripts\download_data.py --dataset mnist cifar10 imdb
```

它们会把数据缓存到 `data/raw/<name>/`，以便后续 `data.load_*` 优先走本地路径。
