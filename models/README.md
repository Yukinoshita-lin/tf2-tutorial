# models/

存放训练好的模型：

- `saved/` — SavedModel 格式，可被 `tensorflow serving` 加载。
- 第 8 章的 `.keras` / `.tflite` 也在本目录（`chapter08_mnist_mlp.*`）。
- 训练中间检查点统一放在 `output/checkpoints/`（按章节命名），避免与本目录混淆。

`.gitignore` 已排除大部分二进制格式，请避免直接提交大文件。
