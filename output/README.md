# output/

训练过程输出：

- `logs/` — TensorBoard 日志 (`fit(..., callbacks=[TensorBoard(...)])`)。
- `checkpoints/` — 训练时自动保存的最佳权重。
- `figures/` 目录下另有训练曲线、混淆矩阵等 PNG 图片。

要查看 TensorBoard：

```powershell
tensorboard --logdir F:\Tensorflow\output\logs
```
